#!/usr/bin/env python3
"""
FlipZip Enhanced Detection Benchmark
=====================================

Tests the new detection methods (period, wavelet) against the original tau
on different signal types.

Author: Mac Mayo
Date: January 2026
"""

import json
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flipzip.core import detect_seams as detect_seams_tau
from flipzip.enhanced import (
    detect_seams_period,
    detect_seams_wavelet,
    detect_seams_adaptive,
    auto_select_detector,
    PYWT_AVAILABLE,
)


# =============================================================================
# TEST SIGNAL GENERATORS
# =============================================================================

def generate_ecg_like(n_samples=10800, fs=360, seed=42):
    """ECG-like with rate change (Normal → Tachycardia → Normal)."""
    np.random.seed(seed)
    
    def qrs_spike(t, width=0.02):
        return np.exp(-((t/width)**2)) * (1 - 2*(t/width)**2)
    
    def make_rhythm(n, fs, rate_hz, rate_var=0.0):
        signal = np.zeros(n)
        t = np.arange(n) / fs
        beat_interval = fs / rate_hz
        pos = 0
        while pos < n:
            bp = int(pos)
            start, end = max(0, bp - int(0.1*fs)), min(n, bp + int(0.1*fs))
            if start < end:
                local_t = (np.arange(start, end) - bp) / fs
                signal[start:end] += qrs_spike(local_t)
            pos += beat_interval * (1 + rate_var * np.random.randn())
            pos = max(pos, bp + fs * 0.3)
        signal += 0.1 * np.sin(2 * np.pi * 0.1 * t)
        signal += 0.02 * np.random.randn(n)
        return signal
    
    n_per = n_samples // 3
    s1 = make_rhythm(n_per, fs, 1.0, 0.05)
    s2 = make_rhythm(n_per, fs, 1.67, 0.15)
    s3 = make_rhythm(n_per, fs, 1.0, 0.05)
    
    return np.concatenate([s1, s2, s3]), [n_per, 2*n_per]


def generate_frequency_switch(n_samples=2000, seed=42):
    """Frequency modulated signal."""
    np.random.seed(seed)
    t = np.linspace(0, 10, n_samples)
    n_per = n_samples // 3
    
    s1 = np.sin(2 * np.pi * 2 * t[:n_per])
    s2 = np.sin(2 * np.pi * 8 * t[n_per:2*n_per])
    s3 = np.sin(2 * np.pi * 2 * t[2*n_per:])
    
    signal = np.concatenate([s1, s2, s3]) + 0.1 * np.random.randn(n_samples)
    return signal, [n_per, 2*n_per]


def generate_amplitude_switch(n_samples=2000, seed=42):
    """Amplitude change (structural change - tau should work)."""
    np.random.seed(seed)
    t = np.linspace(0, 10, n_samples)
    n_per = n_samples // 3
    
    s1 = 1.0 * np.sin(2 * np.pi * 3 * t[:n_per])
    s2 = 0.2 * np.sin(2 * np.pi * 3 * t[n_per:2*n_per])
    s3 = 1.0 * np.sin(2 * np.pi * 3 * t[2*n_per:])
    
    signal = np.concatenate([s1, s2, s3]) + 0.05 * np.random.randn(n_samples)
    return signal, [n_per, 2*n_per]


def generate_piecewise_constant(n_samples=1000, seed=42):
    """Piecewise constant (mean shifts)."""
    np.random.seed(seed)
    n_per = n_samples // 4
    
    s1 = 0.0 + 0.1 * np.random.randn(n_per)
    s2 = 2.0 + 0.1 * np.random.randn(n_per)
    s3 = 1.0 + 0.1 * np.random.randn(n_per)
    s4 = 3.0 + 0.1 * np.random.randn(n_per)
    
    return np.concatenate([s1, s2, s3, s4]), [n_per, 2*n_per, 3*n_per]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_detection(detected, true_points, tolerance):
    """Compute precision, recall, F1."""
    if len(true_points) == 0:
        return {'precision': 1.0 if len(detected) == 0 else 0.0,
                'recall': 1.0, 'f1': 1.0 if len(detected) == 0 else 0.0}
    
    if len(detected) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    tp = 0
    matched = set()
    
    for d in detected:
        for i, t in enumerate(true_points):
            if i not in matched and abs(d - t) <= tolerance:
                tp += 1
                matched.add(i)
                break
    
    precision = tp / len(detected)
    recall = tp / len(true_points)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp}


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_enhanced_benchmark():
    """Run comprehensive benchmark of all detection methods."""
    
    print("=" * 80)
    print("FlipZip Enhanced Detection Benchmark")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 80)
    
    if not PYWT_AVAILABLE:
        print("\nWARNING: PyWavelets not available. Wavelet methods will be skipped.")
    
    # Test cases
    test_cases = [
        ('ecg_like', generate_ecg_like, "Quasi-periodic with rate change"),
        ('frequency_switch', generate_frequency_switch, "Frequency modulation"),
        ('amplitude_switch', generate_amplitude_switch, "Amplitude change"),
        ('piecewise_constant', generate_piecewise_constant, "Mean shifts"),
    ]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'pywt_available': PYWT_AVAILABLE,
        'tests': {}
    }
    
    # Methods to test
    methods = ['tau', 'period']
    if PYWT_AVAILABLE:
        methods.append('wavelet')
    methods.append('adaptive')
    
    for name, generator, description in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {name}")
        print(f"Description: {description}")
        print("=" * 80)
        
        # Generate signal
        if name == 'ecg_like':
            signal, true_points = generator(n_samples=10800)
        else:
            signal, true_points = generator()
        
        n = len(signal)
        tolerance = n // 20
        
        print(f"Signal length: {n}")
        print(f"True change points: {true_points}")
        print(f"Tolerance: ±{tolerance} samples")
        
        # Auto-detection recommendation
        auto_method = auto_select_detector(signal)
        print(f"Auto-selected method: {auto_method}")
        
        test_result = {
            'n_samples': n,
            'true_points': true_points,
            'tolerance': tolerance,
            'auto_selected': auto_method,
            'methods': {}
        }
        
        # Test each method
        for method in methods:
            try:
                if method == 'tau':
                    _, _, seams = detect_seams_tau(signal, window_size=256, stride=128)
                elif method == 'period':
                    _, _, seams = detect_seams_period(signal, window_size=256, stride=128)
                elif method == 'wavelet':
                    _, _, seams = detect_seams_wavelet(signal)
                elif method == 'adaptive':
                    _, seams = detect_seams_adaptive(signal, window_size=256)
                
                eval_result = evaluate_detection(seams, true_points, tolerance)
                
                print(f"\n{method.upper()}:")
                print(f"  Detected: {seams}")
                print(f"  P: {eval_result['precision']:.2f}, R: {eval_result['recall']:.2f}, F1: {eval_result['f1']:.2f}")
                
                test_result['methods'][method] = {
                    'seams': seams,
                    'evaluation': eval_result
                }
                
            except Exception as e:
                print(f"\n{method.upper()}: ERROR - {e}")
                test_result['methods'][method] = {'error': str(e)}
        
        results['tests'][name] = test_result
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: F1 Scores by Method and Signal Type")
    print("=" * 80)
    
    header = f"{'Signal':<25}"
    for m in methods:
        header += f" {m.upper():>10}"
    print(f"\n{header}")
    print("-" * (25 + 11 * len(methods)))
    
    for name in results['tests']:
        row = f"{name:<25}"
        for m in methods:
            if m in results['tests'][name]['methods']:
                method_data = results['tests'][name]['methods'][m]
                if 'evaluation' in method_data:
                    row += f" {method_data['evaluation']['f1']:>10.2f}"
                else:
                    row += f" {'err':>10}"
            else:
                row += f" {'n/a':>10}"
        print(row)
    
    print("-" * (25 + 11 * len(methods)))
    
    # Compute averages
    print("\nAverages:")
    for m in methods:
        f1_scores = []
        for name in results['tests']:
            if m in results['tests'][name]['methods']:
                method_data = results['tests'][name]['methods'][m]
                if 'evaluation' in method_data:
                    f1_scores.append(method_data['evaluation']['f1'])
        if f1_scores:
            print(f"  {m.upper()}: {np.mean(f1_scores):.2f}")
    
    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    output_file = 'enhanced_detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Find best method per signal type
    for name in results['tests']:
        best_method = None
        best_f1 = -1
        for m in methods:
            if m in results['tests'][name]['methods']:
                method_data = results['tests'][name]['methods'][m]
                if 'evaluation' in method_data:
                    f1 = method_data['evaluation']['f1']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_method = m
        print(f"  {name}: Best method = {best_method} (F1={best_f1:.2f})")
    
    return results


if __name__ == '__main__':
    run_enhanced_benchmark()

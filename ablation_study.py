#!/usr/bin/env python3
"""
Change Detection Ablation Study
================================

Compares three approaches to regime change detection:
1. Piecewise (PELT) - classical change-point detection on raw signal
2. Wavelet (DWT) - change detection on wavelet coefficients  
3. Hybrid - PELT on wavelet approximation coefficients

This helps understand whether WHT-based FlipZip's failures on ECG-like
signals could be addressed by wavelet preprocessing.

Author: Mac Mayo
Date: January 2026
"""

import json
import numpy as np
import pywt
import ruptures as rpt
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# SYNTHETIC SIGNAL GENERATORS
# =============================================================================

def generate_ecg_like(n_samples=10800, fs=360, seed=42):
    """
    Generate synthetic ECG-like signal with regime transitions.
    Normal → Tachycardia → Normal
    """
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
            local_t = (np.arange(start, end) - bp) / fs
            signal[start:end] += qrs_spike(local_t)
            pos += beat_interval * (1 + rate_var * np.random.randn())
            pos = max(pos, bp + fs * 0.3)
        signal += 0.1 * np.sin(2 * np.pi * 0.1 * t)
        signal += 0.02 * np.random.randn(n)
        return signal
    
    n_per = n_samples // 3
    s1 = make_rhythm(n_per, fs, 1.0, 0.05)   # Normal 60 bpm
    s2 = make_rhythm(n_per, fs, 1.67, 0.2)   # Tachycardia 100 bpm
    s3 = make_rhythm(n_per, fs, 1.0, 0.05)   # Normal again
    
    return np.concatenate([s1, s2, s3]), [n_per, 2*n_per], fs


def generate_piecewise_constant(n_samples=1000, seed=42):
    """Simple piecewise constant + noise (PELT's ideal case)."""
    np.random.seed(seed)
    n_per = n_samples // 4
    s1 = 0.0 + 0.1 * np.random.randn(n_per)
    s2 = 2.0 + 0.1 * np.random.randn(n_per)
    s3 = 1.0 + 0.1 * np.random.randn(n_per)
    s4 = 3.0 + 0.1 * np.random.randn(n_per)
    return np.concatenate([s1, s2, s3, s4]), [n_per, 2*n_per, 3*n_per]


def generate_frequency_modulated(n_samples=2000, seed=42):
    """Signal with changing frequency (wavelet's ideal case)."""
    np.random.seed(seed)
    t = np.linspace(0, 10, n_samples)
    n_per = n_samples // 3
    
    # Low freq → High freq → Low freq
    s1 = np.sin(2 * np.pi * 1 * t[:n_per])
    s2 = np.sin(2 * np.pi * 5 * t[n_per:2*n_per])
    s3 = np.sin(2 * np.pi * 1 * t[2*n_per:])
    
    signal = np.concatenate([s1, s2, s3]) + 0.1 * np.random.randn(n_samples)
    return signal, [n_per, 2*n_per]


# =============================================================================
# CHANGE DETECTION METHODS
# =============================================================================

def detect_pelt(signal, n_bkps=None, penalty=None, model='rbf'):
    """
    PELT (Pruned Exact Linear Time) change-point detection.
    
    Args:
        signal: 1D array
        n_bkps: Number of breakpoints (if known)
        penalty: Penalty for Pelt (if n_bkps unknown)
        model: 'l2' (mean), 'rbf' (kernel), 'normal' (mean+var)
    
    Returns:
        List of change point indices
    """
    algo = rpt.Pelt(model=model, min_size=10).fit(signal)
    
    if n_bkps is not None:
        # Use Binseg if we know n_bkps (PELT needs penalty)
        algo = rpt.Binseg(model=model, min_size=10).fit(signal)
        result = algo.predict(n_bkps=n_bkps)
    else:
        # Use penalty
        pen = penalty if penalty else np.log(len(signal)) * np.var(signal)
        result = algo.predict(pen=pen)
    
    # Remove the last element (always = len(signal))
    return result[:-1] if result and result[-1] == len(signal) else result


def detect_wavelet_detail(signal, wavelet='db4', level=4, threshold_std=3.0):
    """
    Detect changes via wavelet detail coefficient spikes.
    
    Sharp transitions cause spikes in fine-scale detail coefficients.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Look at finest detail (d1)
    d1 = coeffs[-1]
    
    # Find spikes above threshold
    threshold = threshold_std * np.std(d1)
    spike_idx = np.where(np.abs(d1) > threshold)[0]
    
    # Map back to signal indices (approximate)
    scale_factor = len(signal) / len(d1)
    change_points = (spike_idx * scale_factor).astype(int).tolist()
    
    # Cluster nearby points
    if len(change_points) > 0:
        change_points = cluster_nearby(change_points, min_gap=len(signal)//20)
    
    return change_points


def detect_wavelet_energy(signal, wavelet='db4', level=4, window=50):
    """
    Detect changes via wavelet energy redistribution across scales.
    
    When regime changes, energy shifts between scales.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Compute energy at each scale over sliding windows
    # Use approximation (coarse) for slow changes
    ca = coeffs[0]
    
    # Compute local variance (energy proxy)
    energy = np.array([np.var(ca[max(0,i-window):i+window]) 
                       for i in range(len(ca))])
    
    # Find where energy changes significantly
    energy_diff = np.abs(np.diff(energy))
    threshold = 3 * np.std(energy_diff)
    change_idx = np.where(energy_diff > threshold)[0]
    
    # Map back to signal
    scale_factor = len(signal) / len(ca)
    change_points = (change_idx * scale_factor).astype(int).tolist()
    
    return cluster_nearby(change_points, min_gap=len(signal)//20)


def detect_hybrid_pelt_wavelet(signal, wavelet='db4', level=4, n_bkps=None):
    """
    Hybrid: Apply PELT to wavelet approximation coefficients.
    
    This filters out high-frequency noise before change detection.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    ca = coeffs[0]  # Coarse approximation
    
    # Run PELT on approximation
    change_idx = detect_pelt(ca, n_bkps=n_bkps, model='l2')
    
    # Map back to signal indices
    scale_factor = len(signal) / len(ca)
    change_points = [int(idx * scale_factor) for idx in change_idx]
    
    return change_points


def cluster_nearby(points, min_gap=50):
    """Cluster nearby change points, keeping representative."""
    if len(points) == 0:
        return []
    
    points = sorted(points)
    clusters = [[points[0]]]
    
    for p in points[1:]:
        if p - clusters[-1][-1] < min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    
    # Return median of each cluster
    return [int(np.median(c)) for c in clusters]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_detection(detected, true_points, tolerance=None, n_samples=None):
    """
    Evaluate change point detection accuracy.
    
    Returns precision, recall, F1, and per-point errors.
    """
    if tolerance is None:
        tolerance = n_samples // 20 if n_samples else 50
    
    if len(true_points) == 0:
        return {'precision': 1.0 if len(detected) == 0 else 0.0,
                'recall': 1.0, 'f1': 1.0 if len(detected) == 0 else 0.0}
    
    if len(detected) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Count true positives
    tp = 0
    matched_true = set()
    errors = []
    
    for d in detected:
        best_match = None
        best_dist = float('inf')
        for i, t in enumerate(true_points):
            if i not in matched_true:
                dist = abs(d - t)
                if dist < best_dist:
                    best_dist = dist
                    best_match = i
        
        if best_match is not None and best_dist <= tolerance:
            tp += 1
            matched_true.add(best_match)
            errors.append(best_dist)
    
    precision = tp / len(detected) if detected else 0
    recall = tp / len(true_points)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': len(detected) - tp,
        'false_negatives': len(true_points) - tp,
        'mean_error': np.mean(errors) if errors else None,
    }


# =============================================================================
# MAIN ABLATION STUDY
# =============================================================================

def run_ablation_study():
    """Run complete ablation comparing PELT, Wavelet, and Hybrid methods."""
    
    print("=" * 80)
    print("Change Detection Ablation Study")
    print("Comparing: PELT (piecewise) vs Wavelet vs Hybrid")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Test signals
    test_cases = [
        ('piecewise_constant', generate_piecewise_constant, 
         "Ideal for PELT: abrupt mean shifts"),
        ('frequency_modulated', generate_frequency_modulated,
         "Ideal for Wavelet: frequency changes"),
        ('ecg_like', generate_ecg_like,
         "Challenging: quasi-periodic with rate changes"),
    ]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    for name, generator, description in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {name}")
        print(f"Description: {description}")
        print("=" * 80)
        
        # Generate signal
        if name == 'ecg_like':
            signal, true_points, fs = generator(n_samples=10800)
        else:
            signal, true_points = generator()
        
        n = len(signal)
        tolerance = n // 20  # 5% tolerance
        
        print(f"Signal length: {n}")
        print(f"True change points: {true_points}")
        print(f"Detection tolerance: ±{tolerance} samples")
        
        test_result = {
            'n_samples': n,
            'true_points': true_points,
            'tolerance': tolerance,
            'methods': {}
        }
        
        # Method 1: PELT on raw signal
        print(f"\n--- PELT (raw signal) ---")
        try:
            detected_pelt = detect_pelt(signal, n_bkps=len(true_points))
            eval_pelt = evaluate_detection(detected_pelt, true_points, tolerance, n)
            print(f"Detected: {detected_pelt}")
            print(f"Precision: {eval_pelt['precision']:.2f}, Recall: {eval_pelt['recall']:.2f}, F1: {eval_pelt['f1']:.2f}")
            test_result['methods']['pelt_raw'] = {
                'detected': detected_pelt,
                'evaluation': eval_pelt
            }
        except Exception as e:
            print(f"Error: {e}")
            test_result['methods']['pelt_raw'] = {'error': str(e)}
        
        # Method 2: Wavelet detail spikes
        print(f"\n--- Wavelet Detail (db4, level 4) ---")
        try:
            detected_wav_detail = detect_wavelet_detail(signal, wavelet='db4', level=4)
            eval_wav = evaluate_detection(detected_wav_detail, true_points, tolerance, n)
            print(f"Detected: {detected_wav_detail}")
            print(f"Precision: {eval_wav['precision']:.2f}, Recall: {eval_wav['recall']:.2f}, F1: {eval_wav['f1']:.2f}")
            test_result['methods']['wavelet_detail'] = {
                'detected': detected_wav_detail,
                'evaluation': eval_wav
            }
        except Exception as e:
            print(f"Error: {e}")
            test_result['methods']['wavelet_detail'] = {'error': str(e)}
        
        # Method 3: Wavelet energy
        print(f"\n--- Wavelet Energy ---")
        try:
            detected_wav_energy = detect_wavelet_energy(signal, wavelet='db4', level=4)
            eval_energy = evaluate_detection(detected_wav_energy, true_points, tolerance, n)
            print(f"Detected: {detected_wav_energy}")
            print(f"Precision: {eval_energy['precision']:.2f}, Recall: {eval_energy['recall']:.2f}, F1: {eval_energy['f1']:.2f}")
            test_result['methods']['wavelet_energy'] = {
                'detected': detected_wav_energy,
                'evaluation': eval_energy
            }
        except Exception as e:
            print(f"Error: {e}")
            test_result['methods']['wavelet_energy'] = {'error': str(e)}
        
        # Method 4: Hybrid PELT on wavelet approximation
        print(f"\n--- Hybrid: PELT on Wavelet Approximation ---")
        try:
            detected_hybrid = detect_hybrid_pelt_wavelet(signal, wavelet='db4', level=4, 
                                                         n_bkps=len(true_points))
            eval_hybrid = evaluate_detection(detected_hybrid, true_points, tolerance, n)
            print(f"Detected: {detected_hybrid}")
            print(f"Precision: {eval_hybrid['precision']:.2f}, Recall: {eval_hybrid['recall']:.2f}, F1: {eval_hybrid['f1']:.2f}")
            test_result['methods']['hybrid_pelt_wavelet'] = {
                'detected': detected_hybrid,
                'evaluation': eval_hybrid
            }
        except Exception as e:
            print(f"Error: {e}")
            test_result['methods']['hybrid_pelt_wavelet'] = {'error': str(e)}
        
        results['tests'][name] = test_result
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: F1 Scores by Method and Signal Type")
    print("=" * 80)
    
    print(f"\n{'Signal':<25} {'PELT':>10} {'Wav.Det':>10} {'Wav.Eng':>10} {'Hybrid':>10}")
    print("-" * 70)
    
    for name in results['tests']:
        methods = results['tests'][name]['methods']
        row = [name]
        for m in ['pelt_raw', 'wavelet_detail', 'wavelet_energy', 'hybrid_pelt_wavelet']:
            if m in methods and 'evaluation' in methods[m]:
                row.append(f"{methods[m]['evaluation']['f1']:.2f}")
            else:
                row.append("err")
        print(f"{row[0]:<25} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")
    
    print("-" * 70)
    
    # Save results
    output_file = 'ablation_study_results.json'
    
    # Convert numpy types for JSON
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
    
    with open(output_file, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
If Hybrid (PELT on wavelet approximation) outperforms both:
  → Suggests FlipZip could benefit from wavelet preprocessing
  → WHT alone may not be the right basis for quasi-periodic signals

If PELT works well on ECG but FlipZip's tau fails:
  → Problem is in the sparsity statistic, not the transform choice
  → Tau may need redesign for oscillatory signals
""")
    
    return results


if __name__ == '__main__':
    run_ablation_study()

#!/usr/bin/env python3
"""
FlipZip Synthetic Benchmark
===========================

Reproducible experiments on synthetic regime-switching signals.

This script generates controlled synthetic data with known regime transitions
and evaluates FlipZip's seam detection and compression performance.

Usage:
    python synthetic_benchmark.py

Output:
    - Console: Summary statistics
    - synthetic_results.json: Full results for reproducibility
    - figures/: Visualization plots

Author: Mac Mayo
Date: January 2026
"""

import json
import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flipzip.core import (
    compute_tau_mad,
    compute_tau_energy,
    detect_seams,
    FlipZipCompressor,
    estimate_compression_ratio,
)


# =============================================================================
# CONFIGURATION (LOCKED - DO NOT MODIFY WITHOUT VERSIONING)
# =============================================================================

CONFIG = {
    'version': '1.0.0',
    'random_seed': 42,
    'window_size': 64,
    'stride': 32,
    'quantization_bits': 10,
    'tau_func': 'mad',  # Parameter-free version
    'n_null_permutations': 100,  # For statistical testing
}


# =============================================================================
# SYNTHETIC SIGNAL GENERATORS
# =============================================================================

def generate_sine_to_cosine(
    n_per_regime: int = 256,
    noise_level: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    """
    Synthetic signal: sine wave regime → cosine wave regime.
    
    Ground truth transition at n_per_regime.
    
    Returns:
        signal: Concatenated signal
        transition: Index of true transition
    """
    np.random.seed(seed)
    
    t1 = np.linspace(0, 4 * np.pi, n_per_regime)
    t2 = np.linspace(0, 4 * np.pi, n_per_regime)
    
    regime1 = np.sin(t1) + noise_level * np.random.randn(n_per_regime)
    regime2 = np.cos(t2) + noise_level * np.random.randn(n_per_regime)
    
    signal = np.concatenate([regime1, regime2])
    transition = n_per_regime
    
    return signal, transition


def generate_frequency_switch(
    n_per_regime: int = 256,
    freq1: float = 2.0,
    freq2: float = 8.0,
    noise_level: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    """
    Synthetic signal: low frequency → high frequency.
    
    Returns:
        signal: Concatenated signal
        transition: Index of true transition
    """
    np.random.seed(seed)
    
    t1 = np.linspace(0, 1, n_per_regime)
    t2 = np.linspace(0, 1, n_per_regime)
    
    regime1 = np.sin(2 * np.pi * freq1 * t1) + noise_level * np.random.randn(n_per_regime)
    regime2 = np.sin(2 * np.pi * freq2 * t2) + noise_level * np.random.randn(n_per_regime)
    
    signal = np.concatenate([regime1, regime2])
    transition = n_per_regime
    
    return signal, transition


def generate_amplitude_switch(
    n_per_regime: int = 256,
    amp1: float = 1.0,
    amp2: float = 0.2,
    noise_level: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    """
    Synthetic signal: high amplitude → low amplitude sine.
    
    Returns:
        signal: Concatenated signal
        transition: Index of true transition
    """
    np.random.seed(seed)
    
    t = np.linspace(0, 4 * np.pi, n_per_regime)
    
    regime1 = amp1 * np.sin(t) + noise_level * np.random.randn(n_per_regime)
    regime2 = amp2 * np.sin(t) + noise_level * np.random.randn(n_per_regime)
    
    signal = np.concatenate([regime1, regime2])
    transition = n_per_regime
    
    return signal, transition


def generate_noise_switch(
    n_per_regime: int = 256,
    noise1: float = 0.1,
    noise2: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    """
    Synthetic signal: low noise → high noise (variance regime change).
    
    Returns:
        signal: Concatenated signal
        transition: Index of true transition
    """
    np.random.seed(seed)
    
    regime1 = noise1 * np.random.randn(n_per_regime)
    regime2 = noise2 * np.random.randn(n_per_regime)
    
    signal = np.concatenate([regime1, regime2])
    transition = n_per_regime
    
    return signal, transition


def generate_step_function(
    n_per_regime: int = 256,
    level1: float = 0.0,
    level2: float = 1.0,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    """
    Synthetic signal: step function with noise.
    
    Returns:
        signal: Concatenated signal  
        transition: Index of true transition
    """
    np.random.seed(seed)
    
    regime1 = level1 + noise_level * np.random.randn(n_per_regime)
    regime2 = level2 + noise_level * np.random.randn(n_per_regime)
    
    signal = np.concatenate([regime1, regime2])
    transition = n_per_regime
    
    return signal, transition


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_seam_detection(
    signal: np.ndarray,
    true_transition: int,
    window_size: int = 64,
    stride: int = 32,
    tolerance: int = None
) -> Dict:
    """
    Evaluate seam detection accuracy.
    
    Args:
        signal: Test signal
        true_transition: Ground truth transition index
        tolerance: Detection tolerance in samples (default: stride)
    
    Returns:
        dict with detection results
    """
    if tolerance is None:
        tolerance = stride
    
    positions, tau_values, detected_seams = detect_seams(
        signal,
        window_size=window_size,
        stride=stride,
        tau_func=CONFIG['tau_func']
    )
    
    # Check if any detected seam is within tolerance of true transition
    detected = False
    detection_error = float('inf')
    closest_seam = None
    
    for seam in detected_seams:
        error = abs(seam - true_transition)
        if error < detection_error:
            detection_error = error
            closest_seam = seam
        if error <= tolerance:
            detected = True
    
    # Tau statistics
    tau_at_transition = None
    for i, pos in enumerate(positions):
        if abs(pos - true_transition) <= stride:
            tau_at_transition = tau_values[i]
            break
    
    return {
        'detected': detected,
        'detection_error': detection_error if closest_seam else None,
        'closest_seam': closest_seam,
        'n_seams_detected': len(detected_seams),
        'all_detected_seams': detected_seams,
        'tau_at_transition': tau_at_transition,
        'tau_mean': float(np.mean(tau_values)),
        'tau_std': float(np.std(tau_values)),
        'tau_min': float(np.min(tau_values)),
        'tau_max': float(np.max(tau_values)),
    }


def compute_null_distribution(
    signal: np.ndarray,
    n_permutations: int = 100,
    window_size: int = 64,
    seed: int = 42
) -> np.ndarray:
    """
    Compute null distribution of tau via block permutation.
    
    Destroys regime structure while preserving local autocorrelation.
    """
    np.random.seed(seed)
    
    n = len(signal)
    block_size = window_size
    n_blocks = n // block_size
    
    null_taus = []
    
    for perm in range(n_permutations):
        # Block shuffle
        blocks = [signal[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        np.random.shuffle(blocks)
        shuffled = np.concatenate(blocks)
        
        # Compute tau on shuffled
        tau = compute_tau_mad(shuffled[:window_size])
        null_taus.append(tau)
    
    return np.array(null_taus)


def run_compression_benchmark(
    signal: np.ndarray,
    name: str
) -> Dict:
    """
    Run compression benchmark comparing FlipZip vs baselines.
    """
    # FlipZip
    result = estimate_compression_ratio(
        signal,
        window_size=CONFIG['window_size'],
        quantization_bits=CONFIG['quantization_bits']
    )
    
    # Baseline: raw bytes
    raw_bytes = signal.tobytes()
    raw_bps = 64.0  # float64
    
    # Baseline: simple delta encoding estimate
    delta = np.diff(signal)
    delta_var = np.var(delta)
    signal_var = np.var(signal)
    # Rough estimate: if delta variance is lower, delta coding helps
    delta_ratio = delta_var / (signal_var + 1e-12)
    
    return {
        'name': name,
        'length': len(signal),
        'flipzip_bps': result['bits_per_sample'],
        'flipzip_ratio': result['compression_ratio'],
        'raw_bps': raw_bps,
        'delta_variance_ratio': delta_ratio,
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_all_benchmarks() -> Dict:
    """
    Run complete synthetic benchmark suite.
    
    Returns dict with all results.
    """
    np.random.seed(CONFIG['random_seed'])
    
    print("=" * 70)
    print("FlipZip Synthetic Benchmark")
    print(f"Version: {CONFIG['version']}")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Random seed: {CONFIG['random_seed']}")
    print("=" * 70)
    print()
    
    # Test signals
    test_cases = [
        ('sine_to_cosine', generate_sine_to_cosine),
        ('frequency_switch', generate_frequency_switch),
        ('amplitude_switch', generate_amplitude_switch),
        ('noise_switch', generate_noise_switch),
        ('step_function', generate_step_function),
    ]
    
    results = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    detection_successes = 0
    total_tests = len(test_cases)
    
    for name, generator in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print('='*70)
        
        # Generate signal
        signal, true_transition = generator(seed=CONFIG['random_seed'])
        
        print(f"Signal length: {len(signal)}")
        print(f"True transition at: {true_transition}")
        
        # Seam detection
        detection_result = evaluate_seam_detection(
            signal,
            true_transition,
            window_size=CONFIG['window_size'],
            stride=CONFIG['stride']
        )
        
        if detection_result['detected']:
            detection_successes += 1
            status = "✓ DETECTED"
        else:
            status = "✗ MISSED"
        
        print(f"\nSeam Detection: {status}")
        print(f"  Closest detected seam: {detection_result['closest_seam']}")
        print(f"  Detection error: {detection_result['detection_error']} samples")
        print(f"  Total seams detected: {detection_result['n_seams_detected']}")
        print(f"  τ at transition: {detection_result['tau_at_transition']:.4f}" 
              if detection_result['tau_at_transition'] else "  τ at transition: N/A")
        print(f"  τ mean ± std: {detection_result['tau_mean']:.4f} ± {detection_result['tau_std']:.4f}")
        
        # Null distribution
        null_taus = compute_null_distribution(
            signal,
            n_permutations=CONFIG['n_null_permutations'],
            window_size=CONFIG['window_size'],
            seed=CONFIG['random_seed']
        )
        null_95 = np.percentile(null_taus, 95)
        
        print(f"\nNull Distribution:")
        print(f"  Null 95th percentile: {null_95:.4f}")
        
        if detection_result['tau_at_transition'] is not None:
            separation = detection_result['tau_at_transition'] - null_95
            print(f"  Separation (τ_event - null_95): {separation:.4f}")
        
        # Compression benchmark
        compression_result = run_compression_benchmark(signal, name)
        
        print(f"\nCompression:")
        print(f"  FlipZip bps: {compression_result['flipzip_bps']:.2f}")
        print(f"  Raw bps: {compression_result['raw_bps']:.2f}")
        print(f"  Compression ratio: {compression_result['flipzip_ratio']:.2f}x")
        
        # Store results
        results['tests'][name] = {
            'detection': detection_result,
            'null_95': float(null_95),
            'null_mean': float(np.mean(null_taus)),
            'null_std': float(np.std(null_taus)),
            'compression': compression_result,
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Detection accuracy: {detection_successes}/{total_tests} ({100*detection_successes/total_tests:.1f}%)")
    
    avg_ratio = np.mean([
        results['tests'][name]['compression']['flipzip_ratio'] 
        for name in results['tests']
    ])
    print(f"Average compression ratio: {avg_ratio:.2f}x")
    
    results['summary'] = {
        'detection_accuracy': detection_successes / total_tests,
        'detection_successes': detection_successes,
        'total_tests': total_tests,
        'average_compression_ratio': avg_ratio,
    }
    
    return results


def save_results(results: Dict, output_path: str = 'synthetic_results.json'):
    """Save results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_converted = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    results = run_all_benchmarks()
    save_results(results)
    
    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)

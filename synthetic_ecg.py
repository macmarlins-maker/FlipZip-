#!/usr/bin/env python3
"""
FlipZip Synthetic ECG Benchmark
===============================

Generates synthetic ECG-like signals with regime transitions
(normal → arrhythmic patterns) for benchmarking.

This is used when real PhysioNet data is unavailable due to network restrictions.

Author: Mac Mayo
Date: January 2026
"""

import json
import sys
import os
from datetime import datetime
import gzip
import lzma

import numpy as np

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flipzip.core import (
    FlipZipCompressor,
    detect_seams,
)


def generate_synthetic_ecg(
    n_samples: int = 108000,  # 5 min @ 360 Hz
    fs: float = 360.0,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic ECG-like signal with regime transitions.
    
    Mimics:
    - Normal sinus rhythm (regular QRS complexes)
    - Arrhythmic episodes (irregular patterns)
    
    Returns:
        signal, transition_points
    """
    np.random.seed(seed)
    
    # ECG has ~1 Hz heartbeat frequency with sharp QRS spikes
    # Normal rhythm: regular, ~1 Hz
    # Arrhythmia: irregular timing, altered morphology
    
    def qrs_complex(t, width=0.02):
        """Generate a single QRS-like spike."""
        return np.exp(-((t/width)**2)) * (1 - 2*(t/width)**2)
    
    def generate_rhythm(n, fs, rate_hz, rate_var=0.0):
        """Generate heartbeat rhythm at given rate with variability."""
        signal = np.zeros(n)
        t = np.arange(n) / fs
        
        # Place beats at intervals
        beat_interval = fs / rate_hz  # samples between beats
        beat_positions = []
        
        pos = 0
        while pos < n:
            beat_positions.append(int(pos))
            # Add variability
            interval = beat_interval * (1 + rate_var * np.random.randn())
            pos += max(interval, fs * 0.3)  # minimum 300ms between beats
        
        # Add QRS complexes
        for bp in beat_positions:
            start = max(0, bp - int(0.1 * fs))
            end = min(n, bp + int(0.1 * fs))
            local_t = (np.arange(start, end) - bp) / fs
            signal[start:end] += qrs_complex(local_t, width=0.02)
        
        # Add baseline wander and noise
        signal += 0.1 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz baseline wander
        signal += 0.02 * np.random.randn(n)  # measurement noise
        
        return signal
    
    # Create multi-regime signal
    n_per_regime = n_samples // 3
    
    # Regime 1: Normal sinus rhythm (60 bpm, regular)
    normal1 = generate_rhythm(n_per_regime, fs, rate_hz=1.0, rate_var=0.05)
    
    # Regime 2: Tachycardia (100 bpm, more variable)
    arrhythmic = generate_rhythm(n_per_regime, fs, rate_hz=1.67, rate_var=0.2)
    
    # Regime 3: Back to normal
    normal2 = generate_rhythm(n_per_regime, fs, rate_hz=1.0, rate_var=0.05)
    
    signal = np.concatenate([normal1, arrhythmic, normal2])
    transitions = [n_per_regime, 2 * n_per_regime]
    
    return signal, transitions, fs


def quantize_signal(signal: np.ndarray, bits: int = 10):
    levels = 2 ** bits
    x_min, x_max = signal.min(), signal.max()
    if x_max == x_min:
        return np.zeros(len(signal), dtype=np.int16), x_min, x_max
    scaled = (signal - x_min) / (x_max - x_min) * (levels - 1)
    return np.round(scaled).astype(np.int16), x_min, x_max


def bits_per_sample_gzip(signal: np.ndarray, bits: int = 10):
    quantized, _, _ = quantize_signal(signal, bits)
    compressed = gzip.compress(quantized.tobytes(), compresslevel=9)
    return 8 * len(compressed) / len(signal)


def bits_per_sample_lzma(signal: np.ndarray, bits: int = 10):
    quantized, _, _ = quantize_signal(signal, bits)
    compressed = lzma.compress(quantized.tobytes(), preset=9)
    return 8 * len(compressed) / len(signal)


def bits_per_sample_zstd(signal: np.ndarray, bits: int = 10):
    if not ZSTD_AVAILABLE:
        return None
    quantized, _, _ = quantize_signal(signal, bits)
    cctx = zstd.ZstdCompressor(level=22)
    compressed = cctx.compress(quantized.tobytes())
    return 8 * len(compressed) / len(signal)


def run_synthetic_ecg_benchmark():
    """Run benchmark on synthetic ECG."""
    
    print("=" * 80)
    print("FlipZip Synthetic ECG Benchmark")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 80)
    print("\nGenerating synthetic ECG-like signal with regime transitions...")
    print("  (Normal rhythm → Tachycardia → Normal rhythm)")
    
    signal, transitions, fs = generate_synthetic_ecg(
        n_samples=108000,  # 5 minutes @ 360 Hz
        seed=42
    )
    
    print(f"\nSignal: {len(signal)} samples ({len(signal)/fs:.1f} sec @ {fs} Hz)")
    print(f"Transitions at: {transitions}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'signal_type': 'synthetic_ecg',
        'n_samples': len(signal),
        'fs': fs,
        'transitions': transitions,
    }
    
    # Compression benchmark
    print("\n" + "-" * 80)
    print("COMPRESSION (10-bit quantization, fair comparison)")
    print("-" * 80)
    
    bits = 10
    
    bps_gzip = bits_per_sample_gzip(signal, bits)
    bps_lzma = bits_per_sample_lzma(signal, bits)
    bps_zstd = bits_per_sample_zstd(signal, bits)
    
    compressor = FlipZipCompressor(window_size=256, quantization_bits=bits)
    bps_flipzip = compressor.bits_per_sample(signal)
    
    improvement = (bps_lzma - bps_flipzip) / bps_lzma * 100
    
    print(f"\n{'Method':<12} {'BPS':>10}")
    print("-" * 25)
    print(f"{'GZIP':<12} {bps_gzip:>10.2f}")
    print(f"{'LZMA':<12} {bps_lzma:>10.2f}")
    if bps_zstd:
        print(f"{'zstd':<12} {bps_zstd:>10.2f}")
    print(f"{'FlipZip':<12} {bps_flipzip:>10.2f}")
    print("-" * 25)
    print(f"vs LZMA: {'+' if improvement > 0 else ''}{improvement:.1f}%")
    
    results['compression'] = {
        'bps_gzip': bps_gzip,
        'bps_lzma': bps_lzma,
        'bps_zstd': bps_zstd,
        'bps_flipzip': bps_flipzip,
        'improvement_vs_lzma_pct': improvement,
    }
    
    # Seam detection
    print("\n" + "-" * 80)
    print("SEAM DETECTION")
    print("-" * 80)
    
    positions, tau_values, seams = detect_seams(
        signal, window_size=256, stride=128, tau_func='mad'
    )
    
    print(f"\nSeams detected: {len(seams)}")
    print(f"True transitions: {transitions}")
    
    # Check detection accuracy
    tolerance = 1000  # samples (~2.8 sec)
    detected_transitions = []
    for t in transitions:
        for s in seams:
            if abs(s - t) < tolerance:
                detected_transitions.append(t)
                break
    
    detection_rate = len(detected_transitions) / len(transitions) * 100
    print(f"Detection rate: {len(detected_transitions)}/{len(transitions)} ({detection_rate:.0f}%)")
    
    results['seam_detection'] = {
        'n_seams': len(seams),
        'n_true_transitions': len(transitions),
        'n_detected': len(detected_transitions),
        'detection_rate_pct': detection_rate,
        'tau_mean': float(np.mean(tau_values)),
        'tau_std': float(np.std(tau_values)),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if improvement > 0:
        verdict = f"FlipZip: +{improvement:.1f}% vs LZMA on synthetic ECG"
    else:
        verdict = f"FlipZip: {improvement:.1f}% vs LZMA on synthetic ECG"
    
    print(f"\n{verdict}")
    print(f"Transition detection: {detection_rate:.0f}%")
    
    results['summary'] = {
        'verdict': verdict,
        'improvement_pct': improvement,
        'detection_rate_pct': detection_rate,
    }
    
    # Save
    with open('synthetic_ecg_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: synthetic_ecg_results.json")
    
    return results


if __name__ == '__main__':
    run_synthetic_ecg_benchmark()

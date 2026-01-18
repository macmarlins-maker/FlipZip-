#!/usr/bin/env python3
"""
FlipZip ECG Pilot Benchmark
===========================

Single real-data test using MIT-BIH Arrhythmia Database (PhysioNet).

This is a preliminary pilot on ONE record to establish whether the method
shows any signal on real biomedical data. NOT a comprehensive evaluation.

Usage:
    python ecg_pilot.py

Requires:
    pip install wfdb

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
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("Warning: wfdb not installed. Run: pip install wfdb")

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flipzip.core import (
    FlipZipCompressor,
    detect_seams,
    compute_tau_mad,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'record': '100',           # MIT-BIH record number
    'channel': 0,              # Lead (0 = MLII typically)
    'duration_minutes': 5,     # How much data to use
    'quantization_bits': 10,
    'window_size': 256,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mitbih_record(record: str = '100', duration_minutes: float = 5, channel: int = 0):
    """
    Load a segment from MIT-BIH Arrhythmia Database.
    
    Returns:
        signal: 1D numpy array of ECG samples
        fs: Sampling frequency (Hz)
        annotations: Beat annotations if available
    """
    if not WFDB_AVAILABLE:
        raise RuntimeError("wfdb library not installed")
    
    # Download from PhysioNet if not cached
    print(f"Loading MIT-BIH record {record} from PhysioNet...")
    
    record_data = wfdb.rdrecord(record, pn_dir='mitdb')
    fs = record_data.fs  # Typically 360 Hz
    
    # Calculate samples for requested duration
    n_samples = int(duration_minutes * 60 * fs)
    
    # Extract signal
    signal = record_data.p_signal[:n_samples, channel]
    
    # Remove NaNs if any
    signal = np.nan_to_num(signal, nan=0.0)
    
    # Load annotations (beat labels)
    try:
        ann = wfdb.rdann(record, 'atr', pn_dir='mitdb')
        # Filter annotations to our time window
        ann_samples = ann.sample[ann.sample < n_samples]
        ann_symbols = [ann.symbol[i] for i in range(len(ann_samples))]
    except:
        ann_samples = np.array([])
        ann_symbols = []
    
    print(f"  Loaded {len(signal)} samples ({duration_minutes} min @ {fs} Hz)")
    print(f"  Annotations in window: {len(ann_samples)} beats")
    
    return signal, fs, (ann_samples, ann_symbols)


# =============================================================================
# COMPRESSION BENCHMARKS
# =============================================================================

def quantize_signal(signal: np.ndarray, bits: int = 10):
    """Uniform quantization."""
    levels = 2 ** bits
    x_min, x_max = signal.min(), signal.max()
    if x_max == x_min:
        return np.zeros(len(signal), dtype=np.int16), x_min, x_max
    scaled = (signal - x_min) / (x_max - x_min) * (levels - 1)
    return np.round(scaled).astype(np.int16), x_min, x_max


def bits_per_sample_gzip(signal: np.ndarray, bits: int = 10):
    quantized, x_min, x_max = quantize_signal(signal, bits)
    compressed = gzip.compress(quantized.tobytes(), compresslevel=9)
    return 8 * len(compressed) / len(signal) + 128 / len(signal)


def bits_per_sample_lzma(signal: np.ndarray, bits: int = 10):
    quantized, x_min, x_max = quantize_signal(signal, bits)
    compressed = lzma.compress(quantized.tobytes(), preset=9)
    return 8 * len(compressed) / len(signal) + 128 / len(signal)


def bits_per_sample_zstd(signal: np.ndarray, bits: int = 10):
    if not ZSTD_AVAILABLE:
        return None
    quantized, x_min, x_max = quantize_signal(signal, bits)
    cctx = zstd.ZstdCompressor(level=22)
    compressed = cctx.compress(quantized.tobytes())
    return 8 * len(compressed) / len(signal) + 128 / len(signal)


# =============================================================================
# SEAM ANALYSIS
# =============================================================================

def analyze_seams_vs_annotations(signal, annotations, fs, window_size=256, stride=128):
    """
    Check if detected seams correlate with annotated beat events.
    
    This is exploratory - we don't expect perfect correlation since
    seams detect regime changes, not individual beats.
    """
    ann_samples, ann_symbols = annotations
    
    # Detect seams
    positions, tau_values, seams = detect_seams(
        signal,
        window_size=window_size,
        stride=stride,
        tau_func='mad'
    )
    
    # Find abnormal beats (anything other than N = normal)
    abnormal_indices = [i for i, s in enumerate(ann_symbols) if s not in ['N', '.']]
    abnormal_samples = [ann_samples[i] for i in abnormal_indices]
    
    # Check if any seams are near abnormal beats
    tolerance = 2 * fs  # 2 seconds
    
    seams_near_abnormal = 0
    for seam in seams:
        for ab in abnormal_samples:
            if abs(seam - ab) < tolerance:
                seams_near_abnormal += 1
                break
    
    return {
        'n_seams_detected': len(seams),
        'n_abnormal_beats': len(abnormal_samples),
        'seams_near_abnormal': seams_near_abnormal,
        'tau_mean': float(np.mean(tau_values)),
        'tau_std': float(np.std(tau_values)),
        'tau_min': float(np.min(tau_values)),
        'tau_max': float(np.max(tau_values)),
    }


# =============================================================================
# MAIN PILOT
# =============================================================================

def run_ecg_pilot():
    """Run the ECG pilot benchmark."""
    
    print("=" * 80)
    print("FlipZip ECG Pilot Benchmark")
    print("MIT-BIH Arrhythmia Database (PhysioNet)")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 80)
    print("\nNOTE: This is a PRELIMINARY pilot on ONE record.")
    print("It is NOT a comprehensive evaluation.\n")
    
    if not WFDB_AVAILABLE:
        print("ERROR: wfdb library not available. Install with: pip install wfdb")
        return None
    
    # Load data
    try:
        signal, fs, annotations = load_mitbih_record(
            record=CONFIG['record'],
            duration_minutes=CONFIG['duration_minutes'],
            channel=CONFIG['channel']
        )
    except Exception as e:
        print(f"ERROR loading data: {e}")
        print("This may be a network issue. Try again or check PhysioNet availability.")
        return None
    
    results = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'data': {
            'record': CONFIG['record'],
            'n_samples': len(signal),
            'fs': fs,
            'duration_sec': len(signal) / fs,
        }
    }
    
    # Compression benchmark
    print("\n" + "-" * 80)
    print("COMPRESSION BENCHMARK (Fair comparison at 10-bit quantization)")
    print("-" * 80)
    
    bits = CONFIG['quantization_bits']
    
    bps_gzip = bits_per_sample_gzip(signal, bits)
    bps_lzma = bits_per_sample_lzma(signal, bits)
    bps_zstd = bits_per_sample_zstd(signal, bits) if ZSTD_AVAILABLE else None
    
    compressor = FlipZipCompressor(
        window_size=CONFIG['window_size'],
        quantization_bits=bits
    )
    bps_flipzip = compressor.bits_per_sample(signal)
    
    improvement_vs_lzma = (bps_lzma - bps_flipzip) / bps_lzma * 100
    
    print(f"\n{'Method':<15} {'BPS':>10}")
    print("-" * 30)
    print(f"{'GZIP':<15} {bps_gzip:>10.2f}")
    print(f"{'LZMA':<15} {bps_lzma:>10.2f}")
    if bps_zstd:
        print(f"{'zstd':<15} {bps_zstd:>10.2f}")
    print(f"{'FlipZip':<15} {bps_flipzip:>10.2f}")
    print("-" * 30)
    print(f"FlipZip vs LZMA: {'+' if improvement_vs_lzma > 0 else ''}{improvement_vs_lzma:.1f}%")
    
    results['compression'] = {
        'bps_gzip': bps_gzip,
        'bps_lzma': bps_lzma,
        'bps_zstd': bps_zstd,
        'bps_flipzip': bps_flipzip,
        'improvement_vs_lzma_pct': improvement_vs_lzma,
    }
    
    # Seam analysis
    print("\n" + "-" * 80)
    print("SEAM DETECTION ANALYSIS")
    print("-" * 80)
    
    seam_results = analyze_seams_vs_annotations(
        signal, annotations, fs,
        window_size=CONFIG['window_size']
    )
    
    print(f"\nSeams detected: {seam_results['n_seams_detected']}")
    print(f"Abnormal beats in window: {seam_results['n_abnormal_beats']}")
    print(f"Seams near abnormal beats: {seam_results['seams_near_abnormal']}")
    print(f"τ statistics: {seam_results['tau_mean']:.4f} ± {seam_results['tau_std']:.4f}")
    
    results['seam_analysis'] = seam_results
    
    # Summary
    print("\n" + "=" * 80)
    print("PILOT SUMMARY")
    print("=" * 80)
    
    if improvement_vs_lzma > 0:
        verdict = f"FlipZip shows +{improvement_vs_lzma:.1f}% improvement over LZMA"
    else:
        verdict = f"FlipZip shows {improvement_vs_lzma:.1f}% (worse than LZMA)"
    
    print(f"\n{verdict}")
    print("\nInterpretation:")
    print("  - This is ONE record from ONE database")
    print("  - Results may not generalize")
    print("  - Full evaluation requires multiple records + statistical testing")
    
    results['summary'] = {
        'verdict': verdict,
        'improvement_vs_lzma_pct': improvement_vs_lzma,
    }
    
    # Save results
    output_file = 'ecg_pilot_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    run_ecg_pilot()

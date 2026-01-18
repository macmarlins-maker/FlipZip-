#!/usr/bin/env python3
"""
FlipZip Fair Comparison Benchmark
=================================

Fair comparison: all methods use same quantization level.

This ensures we're comparing:
- FlipZip (WHT + quantization + overhead)
- Baselines (quantization + standard compression)

At the SAME fidelity level.

Author: Mac Mayo
Date: January 2026
"""

import gzip
import lzma
import zstandard as zstd
import numpy as np
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flipzip.core import FlipZipCompressor


def quantize_signal(signal: np.ndarray, bits: int = 10) -> tuple:
    """Uniformly quantize signal to given bit depth."""
    levels = 2 ** bits
    x_min, x_max = signal.min(), signal.max()
    
    if x_max == x_min:
        quantized = np.zeros(len(signal), dtype=np.int16)
    else:
        scaled = (signal - x_min) / (x_max - x_min) * (levels - 1)
        quantized = np.round(scaled).astype(np.int16)
    
    return quantized, x_min, x_max


def bits_per_sample_quantized_gzip(signal: np.ndarray, bits: int = 10) -> tuple:
    """Quantize then GZIP compress."""
    quantized, x_min, x_max = quantize_signal(signal, bits)
    raw_bytes = quantized.tobytes()
    compressed = gzip.compress(raw_bytes, compresslevel=9)
    
    # Total bits = compressed size + header (min/max as float64)
    total_bits = 8 * len(compressed) + 128  # 128 bits for two float64s
    bps = total_bits / len(signal)
    
    # Reconstruction error
    reconstructed = quantized.astype(np.float64) / (2**bits - 1) * (x_max - x_min) + x_min
    mse = np.mean((signal - reconstructed) ** 2)
    
    return bps, mse


def bits_per_sample_quantized_lzma(signal: np.ndarray, bits: int = 10) -> tuple:
    """Quantize then LZMA compress."""
    quantized, x_min, x_max = quantize_signal(signal, bits)
    raw_bytes = quantized.tobytes()
    compressed = lzma.compress(raw_bytes, preset=9)
    
    total_bits = 8 * len(compressed) + 128
    bps = total_bits / len(signal)
    
    reconstructed = quantized.astype(np.float64) / (2**bits - 1) * (x_max - x_min) + x_min
    mse = np.mean((signal - reconstructed) ** 2)
    
    return bps, mse


def bits_per_sample_quantized_zstd(signal: np.ndarray, bits: int = 10) -> tuple:
    """Quantize then zstd compress."""
    quantized, x_min, x_max = quantize_signal(signal, bits)
    raw_bytes = quantized.tobytes()
    cctx = zstd.ZstdCompressor(level=22)
    compressed = cctx.compress(raw_bytes)
    
    total_bits = 8 * len(compressed) + 128
    bps = total_bits / len(signal)
    
    reconstructed = quantized.astype(np.float64) / (2**bits - 1) * (x_max - x_min) + x_min
    mse = np.mean((signal - reconstructed) ** 2)
    
    return bps, mse


def generate_test_signals(seed: int = 42):
    """Generate test signals."""
    np.random.seed(seed)
    signals = {}
    
    # Regime-switching signals (FlipZip target)
    n = 1024
    
    signals['sine_cosine_switch'] = np.concatenate([
        np.sin(np.linspace(0, 8*np.pi, n//2)) + 0.05*np.random.randn(n//2),
        np.cos(np.linspace(0, 8*np.pi, n//2)) + 0.05*np.random.randn(n//2)
    ])
    
    signals['freq_switch_2_to_8hz'] = np.concatenate([
        np.sin(2*np.pi*2*np.linspace(0, 2, n//2)) + 0.05*np.random.randn(n//2),
        np.sin(2*np.pi*8*np.linspace(0, 2, n//2)) + 0.05*np.random.randn(n//2)
    ])
    
    signals['amp_switch'] = np.concatenate([
        np.sin(np.linspace(0, 8*np.pi, n//2)) + 0.05*np.random.randn(n//2),
        0.2*np.sin(np.linspace(0, 8*np.pi, n//2)) + 0.05*np.random.randn(n//2)
    ])
    
    # Control signals
    signals['pure_sine'] = np.sin(np.linspace(0, 16*np.pi, n)) + 0.01*np.random.randn(n)
    signals['white_noise'] = np.random.randn(n)
    signals['random_walk'] = np.cumsum(np.random.randn(n)) / np.sqrt(n)
    
    return signals


def run_fair_comparison(quantization_bits: int = 10):
    """Run fair comparison at specified quantization level."""
    
    print("=" * 90)
    print(f"FlipZip Fair Comparison Benchmark (Quantization: {quantization_bits} bits)")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 90)
    print("\nAll methods use SAME quantization level for fair comparison.")
    print("BPS = bits per sample (lower is better)")
    print()
    
    signals = generate_test_signals(seed=42)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'quantization_bits': quantization_bits,
        'benchmarks': {}
    }
    
    # Table header
    print(f"{'Signal':<25} {'Len':>6} {'GZIP':>8} {'LZMA':>8} {'zstd':>8} {'FlipZip':>8} {'vs LZMA':>9}")
    print("-" * 90)
    
    all_improvements = []
    
    for name, signal in signals.items():
        # Baselines
        bps_gzip, _ = bits_per_sample_quantized_gzip(signal, quantization_bits)
        bps_lzma, _ = bits_per_sample_quantized_lzma(signal, quantization_bits)
        bps_zstd, _ = bits_per_sample_quantized_zstd(signal, quantization_bits)
        
        # FlipZip
        compressor = FlipZipCompressor(
            window_size=256,
            quantization_bits=quantization_bits
        )
        bps_flipzip = compressor.bits_per_sample(signal)
        
        # Improvement vs LZMA
        improvement = (bps_lzma - bps_flipzip) / bps_lzma * 100
        all_improvements.append(improvement)
        
        sign = "+" if improvement > 0 else ""
        
        print(f"{name:<25} {len(signal):>6} {bps_gzip:>8.2f} {bps_lzma:>8.2f} {bps_zstd:>8.2f} {bps_flipzip:>8.2f} {sign}{improvement:>7.1f}%")
        
        results['benchmarks'][name] = {
            'length': len(signal),
            'bps_gzip': float(bps_gzip),
            'bps_lzma': float(bps_lzma),
            'bps_zstd': float(bps_zstd),
            'bps_flipzip': float(bps_flipzip),
            'improvement_vs_lzma_pct': float(improvement),
        }
    
    print("-" * 90)
    
    # Summary
    avg_improvement = np.mean(all_improvements)
    print(f"\nSummary:")
    print(f"  Average vs LZMA: {'+' if avg_improvement > 0 else ''}{avg_improvement:.1f}%")
    print(f"  Best case: {'+' if max(all_improvements) > 0 else ''}{max(all_improvements):.1f}%")
    print(f"  Worst case: {'+' if min(all_improvements) > 0 else ''}{min(all_improvements):.1f}%")
    
    # Regime-switching signals only
    regime_signals = ['sine_cosine_switch', 'freq_switch_2_to_8hz', 'amp_switch']
    regime_improvements = [results['benchmarks'][s]['improvement_vs_lzma_pct'] for s in regime_signals]
    
    print(f"\nRegime-switching signals only:")
    print(f"  Average vs LZMA: {'+' if np.mean(regime_improvements) > 0 else ''}{np.mean(regime_improvements):.1f}%")
    
    results['summary'] = {
        'avg_improvement': float(avg_improvement),
        'regime_avg_improvement': float(np.mean(regime_improvements)),
    }
    
    # Save
    output_file = f'fair_comparison_{quantization_bits}bit.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    print("\n" + "=" * 90)
    print("IMPORTANT: This is a FAIR comparison where all methods use the same quantization.")
    print("Previous benchmarks were misleading because they compared lossy FlipZip to lossless baselines.")
    print("=" * 90)
    
    run_fair_comparison(quantization_bits=10)

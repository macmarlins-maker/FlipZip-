#!/usr/bin/env python3
"""
FlipZip Baseline Comparison Benchmark
=====================================

Compares FlipZip compression against standard baselines:
- GZIP
- LZMA  
- Zstandard (zstd)

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

from flipzip.core import (
    FlipZipCompressor,
    detect_seams,
)


def bits_per_sample_gzip(signal: np.ndarray, level: int = 9) -> float:
    """Compress with gzip and return bps."""
    raw_bytes = signal.astype(np.float64).tobytes()
    compressed = gzip.compress(raw_bytes, compresslevel=level)
    return 8 * len(compressed) / len(signal)


def bits_per_sample_lzma(signal: np.ndarray, preset: int = 9) -> float:
    """Compress with LZMA and return bps."""
    raw_bytes = signal.astype(np.float64).tobytes()
    compressed = lzma.compress(raw_bytes, preset=preset)
    return 8 * len(compressed) / len(signal)


def bits_per_sample_zstd(signal: np.ndarray, level: int = 22) -> float:
    """Compress with zstd and return bps."""
    raw_bytes = signal.astype(np.float64).tobytes()
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(raw_bytes)
    return 8 * len(compressed) / len(signal)


def generate_test_signals(seed: int = 42):
    """Generate suite of test signals."""
    np.random.seed(seed)
    
    signals = {}
    
    # 1. Sine to cosine (phase shift)
    n = 512
    signals['sine_to_cosine'] = {
        'data': np.concatenate([
            np.sin(np.linspace(0, 4*np.pi, n//2)) + 0.05*np.random.randn(n//2),
            np.cos(np.linspace(0, 4*np.pi, n//2)) + 0.05*np.random.randn(n//2)
        ]),
        'transition': n//2
    }
    
    # 2. Frequency switch
    signals['frequency_switch'] = {
        'data': np.concatenate([
            np.sin(2*np.pi*2*np.linspace(0, 1, n//2)) + 0.05*np.random.randn(n//2),
            np.sin(2*np.pi*8*np.linspace(0, 1, n//2)) + 0.05*np.random.randn(n//2)
        ]),
        'transition': n//2
    }
    
    # 3. Multi-regime (3 regimes)
    n3 = 768
    signals['multi_regime'] = {
        'data': np.concatenate([
            np.sin(np.linspace(0, 4*np.pi, n3//3)) + 0.05*np.random.randn(n3//3),
            0.5 + 0.1*np.random.randn(n3//3),  # Flat noisy
            np.sin(np.linspace(0, 8*np.pi, n3//3)) + 0.05*np.random.randn(n3//3)
        ]),
        'transitions': [n3//3, 2*n3//3]
    }
    
    # 4. Longer signal (more realistic length)
    n_long = 4096
    signals['long_signal'] = {
        'data': np.concatenate([
            np.sin(np.linspace(0, 20*np.pi, n_long//2)) + 0.05*np.random.randn(n_long//2),
            np.cos(np.linspace(0, 40*np.pi, n_long//2)) + 0.05*np.random.randn(n_long//2)
        ]),
        'transition': n_long//2
    }
    
    # 5. Pure noise (control - should be hardest to compress)
    signals['pure_noise'] = {
        'data': np.random.randn(512),
        'transition': None
    }
    
    # 6. Highly structured (should be easiest to compress)
    t = np.linspace(0, 4*np.pi, 512)
    signals['structured'] = {
        'data': np.sin(t) + 0.5*np.sin(3*t) + 0.25*np.sin(5*t) + 0.01*np.random.randn(512),
        'transition': None
    }
    
    return signals


def run_baseline_comparison():
    """Run full baseline comparison."""
    
    print("=" * 80)
    print("FlipZip Baseline Comparison Benchmark")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 80)
    
    signals = generate_test_signals(seed=42)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }
    
    # Table header
    print(f"\n{'Signal':<20} {'Length':>8} {'GZIP':>10} {'LZMA':>10} {'zstd':>10} {'FlipZip':>10} {'vs LZMA':>10}")
    print("-" * 80)
    
    for name, sig_info in signals.items():
        signal = sig_info['data']
        
        # Compute bps for each method
        bps_gzip = bits_per_sample_gzip(signal)
        bps_lzma = bits_per_sample_lzma(signal)
        bps_zstd = bits_per_sample_zstd(signal)
        
        # FlipZip
        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)
        bps_flipzip = compressor.bits_per_sample(signal)
        
        # Comparison vs LZMA
        vs_lzma = (bps_lzma - bps_flipzip) / bps_lzma * 100
        
        print(f"{name:<20} {len(signal):>8} {bps_gzip:>10.2f} {bps_lzma:>10.2f} {bps_zstd:>10.2f} {bps_flipzip:>10.2f} {vs_lzma:>9.1f}%")
        
        results['benchmarks'][name] = {
            'length': len(signal),
            'bps_gzip': bps_gzip,
            'bps_lzma': bps_lzma,
            'bps_zstd': bps_zstd,
            'bps_flipzip': bps_flipzip,
            'improvement_vs_lzma_pct': vs_lzma,
        }
    
    print("-" * 80)
    
    # Summary statistics
    improvements = [r['improvement_vs_lzma_pct'] for r in results['benchmarks'].values()]
    avg_improvement = np.mean(improvements)
    
    print(f"\nAverage improvement vs LZMA: {avg_improvement:.1f}%")
    print(f"Best case: {max(improvements):.1f}%")
    print(f"Worst case: {min(improvements):.1f}%")
    
    results['summary'] = {
        'avg_improvement_vs_lzma': avg_improvement,
        'best_improvement': max(improvements),
        'worst_improvement': min(improvements),
    }
    
    # Save results
    with open('baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: baseline_comparison.json")
    
    return results


if __name__ == '__main__':
    run_baseline_comparison()

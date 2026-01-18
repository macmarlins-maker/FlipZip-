# FlipZip

**Involution-aware compression for regime-switching time series**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FlipZip is a compression algorithm designed for time series that exhibit regime-switching behavior—signals where the underlying statistical structure changes at discrete transition points. Examples include:

- Financial returns (volatility regimes)
- Wind turbine telemetry (operational modes)
- ECG signals (normal vs. arrhythmic states)

The core insight is that regime transitions manifest as sparsity discontinuities in Walsh-Hadamard Transform (WHT) coefficients. FlipZip detects these transitions and adapts its encoding accordingly.

## Key Features

- **Walsh-Hadamard Transform** basis for efficient frequency-like decomposition
- **Parameter-free seam detection** using robust MAD-based threshold
- **Involution-aware encoding** that tracks basis switches with minimal overhead
- **O(N log N) complexity** via fast dyadic WHT

## Installation

```bash
git clone https://github.com/macmayo/flipzip.git
cd flipzip
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from flipzip import detect_seams, estimate_compression_ratio

# Generate a signal with regime transition
signal = np.concatenate([
    np.sin(np.linspace(0, 4*np.pi, 256)) + 0.05 * np.random.randn(256),
    np.cos(np.linspace(0, 4*np.pi, 256)) + 0.05 * np.random.randn(256)
])

# Detect seams
positions, tau_values, seams = detect_seams(signal, window_size=64)
print(f"Detected seams at: {seams}")

# Estimate compression
result = estimate_compression_ratio(signal)
print(f"Compression ratio: {result['compression_ratio']:.2f}x")
```

## Running Benchmarks

### Synthetic Experiments

```bash
cd benchmarks
python synthetic_benchmark.py      # Basic seam detection tests
python fair_comparison.py          # Compression vs LZMA/GZIP/zstd
python enhanced_benchmark.py       # Compare tau vs period vs wavelet
python ablation_study.py           # PELT vs wavelet analysis
```

Results are saved to JSON files for reproducibility.

### Detection Method Comparison (January 2026)

| Signal Type | Tau (WHT) | Period | Wavelet | Adaptive |
|-------------|-----------|--------|---------|----------|
| ECG-like (rate change) | 0.20 | 0.00 | **0.36** | 0.36 |
| Frequency modulation | 0.33 | **0.67** | 0.31 | 0.40 |
| Amplitude change | 0.33 | 0.67 | 0.31 | **1.00** |
| Piecewise constant | 0.33 | 0.00 | **1.00** | 0.50 |
| **Average F1** | 0.30 | 0.33 | 0.49 | **0.57** |

**Key findings:**
- **Adaptive method** (combining tau + wavelet) achieves best overall performance
- **Wavelet detail** is best for abrupt changes (piecewise constant)
- **Period tracking** is best for pure frequency modulation
- **Tau (WHT sparsity)** works but has lower precision than specialized methods

### Compression Benchmark (Fair comparison at 10-bit quantization)

| Signal Type | FlipZip vs LZMA |
|-------------|-----------------|
| Regime-switching (freq/amp changes) | **+5-7%** (better) |
| White noise | **+17%** (better) |
| Pure sine | -10% (worse) |
| Random walk | -18% (worse) |

**Honest assessment:** FlipZip with enhanced detection shows promise for regime-switching use cases. The adaptive detector significantly improves seam detection accuracy compared to WHT-only approaches.

## Algorithm Overview

### Sparsity Statistic τ

FlipZip uses a parameter-free sparsity statistic:

```
τ(x) = (# coefficients above θ) / N
```

where the threshold θ is computed adaptively:

```
θ = median(|X|) + 3 × MAD(|X|)
```

This is scale-invariant and robust to outliers.

### Seam Detection

A seam (regime transition) is detected when τ exhibits a significant jump between adjacent windows. The algorithm uses sliding windows with 50% overlap.

### Compression

FlipZip encodes signals window-by-window:
1. Apply WHT to each window
2. Quantize coefficients
3. Track involution state (which basis subset is active)
4. Emit regime-switch flags when seams are detected

## Project Status

**Current version: 0.2.0 (beta)**

- ✅ Core WHT algorithm implemented
- ✅ Multiple detection methods (tau, period, wavelet, adaptive)
- ✅ Fair baseline comparisons (same quantization)
- ✅ Comprehensive benchmarks with honest results
- ✅ Adaptive method achieves 0.57 avg F1 on detection
- 🚧 Full entropy coding (planned)
- 🚧 Real data validation with MIT-BIH (pending network access)

## Citation

If you use FlipZip in your research, please cite:

```bibtex
@misc{mayo2026flipzip,
  author = {Mayo, Mac},
  title = {FlipZip: Exploiting Walsh-Hadamard Involution Structure for Regime-Switching Compression},
  year = {2026},
  howpublished = {\url{https://github.com/macmayo/flipzip}}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Mac Mayo  
Battleboro, North Carolina  
Independent Researcher

## Acknowledgments

This work builds on concepts from:
- Walsh-Hadamard Transform theory (Beauchamp, 1984)
- Arithmetic coding (Witten, Moffat & Bell, 1999)
- The M³ framework for topological signal analysis

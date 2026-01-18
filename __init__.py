"""
FlipZip: Involution-aware compression for regime-switching time series
======================================================================

Author: Mac Mayo
License: MIT
Repository: github.com/macmayo/flipzip
"""

from .core import (
    fast_walsh_hadamard,
    inverse_walsh_hadamard,
    compute_tau_mad,
    compute_tau_energy,
    detect_seams,
    FlipZipCompressor,
    estimate_compression_ratio,
)

from .enhanced import (
    estimate_period_autocorr,
    compute_period_change,
    detect_seams_period,
    detect_seams_wavelet,
    detect_seams_wavelet_energy,
    detect_seams_adaptive,
    auto_select_detector,
    FlipZipEnhanced,
)

__version__ = "0.2.0"
__author__ = "Mac Mayo"

__all__ = [
    'fast_walsh_hadamard',
    'inverse_walsh_hadamard',
    'compute_tau_mad',
    'compute_tau_energy',
    'detect_seams',
    'FlipZipCompressor',
    'estimate_compression_ratio',
]

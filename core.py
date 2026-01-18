"""
FlipZip Core Module
====================
Walsh-Hadamard Transform based compression with involution-aware regime detection.

Author: Mac Mayo
License: MIT
"""

import numpy as np
from typing import Tuple, List, Optional


def fast_walsh_hadamard(x: np.ndarray) -> np.ndarray:
    """
    Compute the Walsh-Hadamard Transform using fast dyadic recursion.
    
    Input length must be a power of 2.
    Returns normalized WHT coefficients.
    
    Complexity: O(N log N)
    """
    n = len(x)
    if n == 0:
        return x
    if n & (n - 1) != 0:
        raise ValueError(f"Input length must be power of 2, got {n}")
    
    # Copy to avoid modifying input
    result = x.astype(np.float64).copy()
    
    # Dyadic recursion (in-place)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = result[j]
                b = result[j + h]
                result[j] = a + b
                result[j + h] = a - b
        h *= 2
    
    # Normalize
    return result / np.sqrt(n)


def inverse_walsh_hadamard(X: np.ndarray) -> np.ndarray:
    """
    Inverse WHT (same as forward due to self-inverse property).
    """
    return fast_walsh_hadamard(X)


def compute_tau_mad(x: np.ndarray) -> float:
    """
    Compute the parameter-free sparsity statistic tau.
    
    tau = (# coefficients above adaptive threshold) / N
    
    Threshold is computed as:
        theta = median(|X|) + 3 * MAD(|X|)
    
    where MAD = median absolute deviation (robust scale estimator).
    
    This is scale-invariant and parameter-free.
    
    Returns:
        tau: float in [0, 1], normalized sparsity fraction
    """
    # Ensure power of 2
    n = len(x)
    if n & (n - 1) != 0:
        # Pad to next power of 2
        next_pow2 = 1 << (n - 1).bit_length()
        x_padded = np.zeros(next_pow2)
        x_padded[:n] = x
        x = x_padded
        n = next_pow2
    
    # Compute WHT
    X = fast_walsh_hadamard(x)
    abs_X = np.abs(X)
    
    # Robust threshold using MAD
    median_X = np.median(abs_X)
    mad = np.median(np.abs(abs_X - median_X))
    # Scale factor 1.4826 makes MAD consistent with std for Gaussian
    robust_scale = 1.4826 * mad
    
    theta = median_X + 3 * robust_scale
    
    # Count coefficients above threshold
    n_significant = np.sum(abs_X > theta)
    tau = n_significant / n
    
    return tau


def compute_tau_energy(x: np.ndarray, target_energy: float = 0.721) -> float:
    """
    Compute sparsity statistic based on energy concentration.
    
    tau = (# coefficients needed for target_energy fraction) / N
    
    This version uses the target energy threshold (default k* = 0.721).
    
    Returns:
        tau: float in [0, 1]
    """
    # Ensure power of 2
    n = len(x)
    if n & (n - 1) != 0:
        next_pow2 = 1 << (n - 1).bit_length()
        x_padded = np.zeros(next_pow2)
        x_padded[:n] = x
        x = x_padded
        n = next_pow2
    
    # Compute WHT
    X = fast_walsh_hadamard(x)
    
    # Sort coefficients by magnitude (descending)
    sorted_sq = np.sort(X**2)[::-1]
    total_energy = np.sum(sorted_sq) + 1e-12  # avoid division by zero
    
    # Find how many coefficients needed for target_energy
    cumsum = np.cumsum(sorted_sq)
    threshold_energy = target_energy * total_energy
    
    n_needed = np.searchsorted(cumsum, threshold_energy) + 1
    tau = n_needed / n
    
    return tau


def detect_seams(
    signal: np.ndarray,
    window_size: int = 64,
    stride: int = 32,
    tau_func: str = 'mad',
    tau_crit: Optional[float] = None,
    sensitivity: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Detect regime seams in a signal using sliding window tau analysis.
    
    Args:
        signal: 1D time series
        window_size: Size of sliding window (should be power of 2)
        stride: Step size between windows
        tau_func: 'mad' for parameter-free, 'energy' for k*-based
        tau_crit: Critical threshold for seam detection (auto-calibrated if None)
        sensitivity: Multiplier for change detection threshold (lower = more sensitive)
    
    Returns:
        positions: Center positions of each window
        tau_values: Tau value at each position
        seam_indices: Detected seam positions (window centers)
    """
    n = len(signal)
    
    # Auto-adjust window size if too large for signal
    if window_size > n // 4:
        window_size = max(32, n // 8)
        window_size = 1 << (window_size - 1).bit_length()  # Round to power of 2
        stride = window_size // 2
    
    positions = []
    tau_values = []
    
    # Compute tau function
    compute_tau = compute_tau_mad if tau_func == 'mad' else compute_tau_energy
    
    for start in range(0, n - window_size + 1, stride):
        window = signal[start:start + window_size]
        tau = compute_tau(window)
        positions.append(start + window_size // 2)
        tau_values.append(tau)
    
    positions = np.array(positions)
    tau_values = np.array(tau_values)
    
    if len(tau_values) < 3:
        return positions, tau_values, []
    
    # Detect seams using multiple criteria
    seam_set = set()
    
    # Criterion 1: Large jumps in tau (derivative-based)
    tau_diff = np.abs(np.diff(tau_values))
    if len(tau_diff) > 0 and np.std(tau_diff) > 0:
        jump_threshold = np.mean(tau_diff) + sensitivity * np.std(tau_diff)
        jumps = positions[1:][tau_diff > jump_threshold]
        seam_set.update(jumps.tolist())
    
    # Criterion 2: Local maxima in tau that exceed threshold
    if tau_crit is None:
        tau_crit = np.mean(tau_values) + sensitivity * np.std(tau_values)
    
    for i in range(1, len(tau_values) - 1):
        # Local maximum above threshold
        if (tau_values[i] > tau_values[i-1] and 
            tau_values[i] > tau_values[i+1] and
            tau_values[i] > tau_crit):
            seam_set.add(positions[i])
    
    # Criterion 3: Transitions from low to high or high to low
    median_tau = np.median(tau_values)
    above_median = tau_values > median_tau
    transitions = np.diff(above_median.astype(int))
    seam_set.update(positions[1:][transitions != 0].tolist())
    
    # Sort and cluster nearby detections
    all_seams = sorted(seam_set)
    
    # Cluster seams that are too close together
    if len(all_seams) > 1:
        clustered = [all_seams[0]]
        for s in all_seams[1:]:
            if s - clustered[-1] > stride * 2:
                clustered.append(s)
            # Otherwise merge by keeping the one with higher tau change
        all_seams = clustered
    
    return positions, tau_values, all_seams


class FlipZipCompressor:
    """
    FlipZip compression algorithm.
    
    Uses WHT + involution-aware encoding for regime-switching signals.
    """
    
    def __init__(
        self,
        window_size: int = 256,
        quantization_bits: int = 10,
        tau_func: str = 'mad'
    ):
        self.window_size = window_size
        self.quantization_bits = quantization_bits
        self.tau_func = tau_func
        self.levels = 2 ** quantization_bits
    
    def _quantize(self, X: np.ndarray) -> np.ndarray:
        """Uniform quantization of WHT coefficients."""
        x_min, x_max = X.min(), X.max()
        if x_max == x_min:
            return np.zeros_like(X, dtype=np.int32)
        
        # Scale to [0, levels-1]
        scaled = (X - x_min) / (x_max - x_min) * (self.levels - 1)
        return np.round(scaled).astype(np.int32)
    
    def _dequantize(self, Q: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
        """Inverse quantization."""
        return Q / (self.levels - 1) * (x_max - x_min) + x_min
    
    def encode(self, signal: np.ndarray) -> dict:
        """
        Encode signal using FlipZip.
        
        Returns dict with compressed representation.
        (This is a simplified version for benchmarking - 
         full entropy coding not yet implemented)
        """
        n = len(signal)
        n_windows = (n + self.window_size - 1) // self.window_size
        
        # Pad to multiple of window_size
        padded_len = n_windows * self.window_size
        padded = np.zeros(padded_len)
        padded[:n] = signal
        
        encoded_windows = []
        seam_flags = []
        
        compute_tau = compute_tau_mad if self.tau_func == 'mad' else compute_tau_energy
        prev_tau = None
        
        for i in range(n_windows):
            start = i * self.window_size
            end = start + self.window_size
            window = padded[start:end]
            
            # Transform
            X = fast_walsh_hadamard(window)
            tau = compute_tau(window)
            
            # Detect seam (tau change)
            is_seam = prev_tau is not None and abs(tau - prev_tau) > 0.1
            seam_flags.append(is_seam)
            prev_tau = tau
            
            # Quantize
            Q = self._quantize(X)
            
            encoded_windows.append({
                'quantized': Q,
                'min': X.min(),
                'max': X.max(),
                'tau': tau
            })
        
        return {
            'windows': encoded_windows,
            'seam_flags': seam_flags,
            'original_length': n,
            'window_size': self.window_size
        }
    
    def decode(self, encoded: dict) -> np.ndarray:
        """Decode FlipZip encoded signal."""
        windows = encoded['windows']
        original_length = encoded['original_length']
        window_size = encoded['window_size']
        
        reconstructed = []
        
        for w in windows:
            X_approx = self._dequantize(w['quantized'], w['min'], w['max'])
            window = inverse_walsh_hadamard(X_approx)
            reconstructed.extend(window)
        
        return np.array(reconstructed[:original_length])
    
    def bits_per_sample(self, signal: np.ndarray) -> float:
        """
        Estimate bits per sample for the signal.
        
        This is a simplified estimate based on quantization + overhead.
        Full implementation would use actual entropy coding.
        """
        encoded = self.encode(signal)
        
        # Count bits
        n_windows = len(encoded['windows'])
        
        # Quantized coefficients: quantization_bits per coefficient
        coeff_bits = n_windows * self.window_size * self.quantization_bits
        
        # Min/max per window: 64 bits each (float64)
        range_bits = n_windows * 2 * 64
        
        # Seam flags: 1 bit each
        seam_bits = n_windows
        
        # Header overhead
        header_bits = 64  # original_length, window_size, etc.
        
        total_bits = coeff_bits + range_bits + seam_bits + header_bits
        
        return total_bits / len(signal)


# Convenience function for quick compression ratio estimation
def estimate_compression_ratio(
    signal: np.ndarray,
    window_size: int = 256,
    quantization_bits: int = 10
) -> dict:
    """
    Estimate compression metrics for a signal.
    
    Returns dict with bits_per_sample and compression_ratio.
    """
    compressor = FlipZipCompressor(
        window_size=window_size,
        quantization_bits=quantization_bits
    )
    
    bps = compressor.bits_per_sample(signal)
    
    # Original: 64 bits per sample (float64)
    original_bps = 64.0
    
    return {
        'bits_per_sample': bps,
        'compression_ratio': original_bps / bps,
        'original_bps': original_bps
    }

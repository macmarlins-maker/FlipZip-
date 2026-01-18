"""
FlipZip Enhanced Detection Module
==================================

Additional regime detection methods for oscillatory signals:
1. Period tracking (autocorrelation-based) - for rate changes
2. Wavelet detail coefficients - for abrupt transitions
3. Adaptive method selection - auto-chooses best detector

These address the limitation that WHT sparsity (tau) fails on 
quasi-periodic signals like ECG where rate changes don't alter sparsity.

Author: Mac Mayo
Date: January 2026
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy import signal as scipy_signal

# Try to import pywt, fallback gracefully
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


# =============================================================================
# PERIOD / RATE TRACKING
# =============================================================================

def estimate_period_autocorr(x: np.ndarray, min_period: int = 10, max_period: int = None) -> float:
    """
    Estimate dominant period using autocorrelation.
    
    Args:
        x: Signal window
        min_period: Minimum period to search (samples)
        max_period: Maximum period to search (default: len(x)//2)
    
    Returns:
        Estimated period in samples (0 if no clear period found)
    """
    n = len(x)
    if max_period is None:
        max_period = n // 2
    
    # Normalize
    x_norm = x - np.mean(x)
    var = np.var(x_norm)
    if var < 1e-10:
        return 0.0
    
    # Compute autocorrelation for lags in [min_period, max_period]
    autocorr = np.correlate(x_norm, x_norm, mode='full')
    autocorr = autocorr[n-1:]  # Keep positive lags only
    autocorr = autocorr / (var * n)  # Normalize
    
    # Find first significant peak after min_period
    search_range = autocorr[min_period:max_period]
    if len(search_range) == 0:
        return 0.0
    
    # Find peaks
    peaks = []
    for i in range(1, len(search_range) - 1):
        if search_range[i] > search_range[i-1] and search_range[i] > search_range[i+1]:
            if search_range[i] > 0.3:  # Threshold for significant correlation
                peaks.append((i + min_period, search_range[i]))
    
    if not peaks:
        return 0.0
    
    # Return period of highest peak
    best_peak = max(peaks, key=lambda p: p[1])
    return float(best_peak[0])


def compute_period_change(
    signal: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
    min_period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track period changes across a signal.
    
    Returns:
        positions: Center of each window
        periods: Estimated period at each position
        period_changes: Absolute change in period between windows
    """
    n = len(signal)
    positions = []
    periods = []
    
    for start in range(0, n - window_size + 1, stride):
        window = signal[start:start + window_size]
        period = estimate_period_autocorr(window, min_period=min_period)
        positions.append(start + window_size // 2)
        periods.append(period)
    
    positions = np.array(positions)
    periods = np.array(periods)
    
    # Compute period changes
    period_changes = np.zeros(len(periods))
    period_changes[1:] = np.abs(np.diff(periods))
    
    return positions, periods, period_changes


def detect_seams_period(
    signal: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
    threshold_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Detect regime seams based on period/rate changes.
    
    This works for quasi-periodic signals like ECG where the fundamental
    period (heart rate) changes between regimes.
    
    Returns:
        positions: Window centers
        period_changes: Period change magnitude at each position
        seams: Detected seam indices
    """
    positions, periods, period_changes = compute_period_change(
        signal, window_size, stride
    )
    
    # Threshold: mean + threshold_std * std of changes
    threshold = np.mean(period_changes) + threshold_std * np.std(period_changes)
    
    # Find significant changes
    seam_mask = period_changes > threshold
    seams = positions[seam_mask].tolist()
    
    # Cluster nearby detections
    seams = _cluster_nearby(seams, min_gap=window_size)
    
    return positions, period_changes, seams


# =============================================================================
# WAVELET-BASED DETECTION
# =============================================================================

def compute_wavelet_detail(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 4
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute wavelet decomposition and return detail coefficients.
    
    Returns:
        approximation: Coarse approximation coefficients
        details: List of detail coefficients [d1, d2, ..., dN] (finest to coarsest)
    """
    if not PYWT_AVAILABLE:
        raise RuntimeError("PyWavelets not installed. Run: pip install PyWavelets")
    
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approximation = coeffs[0]
    details = coeffs[1:][::-1]  # Reverse to get finest first
    
    return approximation, details


def detect_seams_wavelet(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 4,
    threshold_std: float = 3.0,
    use_detail_level: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Detect regime seams using wavelet detail coefficient spikes.
    
    Sharp transitions cause spikes in fine-scale detail coefficients.
    
    Args:
        signal: Input signal
        wavelet: Wavelet family ('db4', 'sym4', 'haar', etc.)
        level: Decomposition level
        threshold_std: Threshold in standard deviations
        use_detail_level: Which detail level to use (1=finest, 2=next, etc.)
    
    Returns:
        positions: Sample positions (mapped from wavelet indices)
        detail_magnitude: |detail coefficients| at each position
        seams: Detected seam indices in signal coordinates
    """
    if not PYWT_AVAILABLE:
        raise RuntimeError("PyWavelets not installed")
    
    approx, details = compute_wavelet_detail(signal, wavelet, level)
    
    # Use specified detail level
    detail_idx = min(use_detail_level - 1, len(details) - 1)
    detail = details[detail_idx]
    
    abs_detail = np.abs(detail)
    
    # Map detail indices to signal indices
    scale_factor = len(signal) / len(detail)
    positions = np.arange(len(detail)) * scale_factor + scale_factor / 2
    
    # Threshold
    threshold = np.mean(abs_detail) + threshold_std * np.std(abs_detail)
    
    # Find spikes
    spike_mask = abs_detail > threshold
    seam_indices_detail = np.where(spike_mask)[0]
    
    # Map to signal coordinates
    seams = [int(idx * scale_factor + scale_factor / 2) for idx in seam_indices_detail]
    
    # Cluster nearby
    seams = _cluster_nearby(seams, min_gap=int(scale_factor * 5))
    
    return positions, abs_detail, seams


def detect_seams_wavelet_energy(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 4,
    window: int = 10,
    threshold_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Detect regime changes via wavelet energy redistribution.
    
    Monitors how energy shifts between scales - regime changes often
    cause energy to redistribute across the wavelet decomposition.
    
    Returns:
        positions: Sample positions
        energy_change: Energy change magnitude at each position
        seams: Detected seam indices
    """
    if not PYWT_AVAILABLE:
        raise RuntimeError("PyWavelets not installed")
    
    approx, details = compute_wavelet_detail(signal, wavelet, level)
    
    # Compute local energy in approximation (tracks slow regime shifts)
    n = len(approx)
    energy = np.array([
        np.var(approx[max(0, i-window):min(n, i+window)])
        for i in range(n)
    ])
    
    # Compute energy gradient
    energy_change = np.zeros(n)
    energy_change[1:] = np.abs(np.diff(energy))
    
    # Map to signal coordinates
    scale_factor = len(signal) / n
    positions = np.arange(n) * scale_factor + scale_factor / 2
    
    # Threshold
    threshold = np.mean(energy_change) + threshold_std * np.std(energy_change)
    
    # Find significant changes
    seam_mask = energy_change > threshold
    seams = [int(positions[i]) for i in range(n) if seam_mask[i]]
    
    # Cluster
    seams = _cluster_nearby(seams, min_gap=int(scale_factor * 10))
    
    return positions, energy_change, seams


# =============================================================================
# ADAPTIVE / COMBINED DETECTION
# =============================================================================

def detect_seams_adaptive(
    signal: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
    methods: List[str] = None
) -> Tuple[dict, List[int]]:
    """
    Adaptive seam detection using multiple methods.
    
    Combines results from different detectors and uses voting/consensus.
    
    Args:
        signal: Input signal
        window_size: Window size for analysis
        stride: Stride between windows
        methods: List of methods to use. Options:
            - 'period': Period/rate tracking (good for oscillatory)
            - 'wavelet': Wavelet detail spikes (good for abrupt changes)
            - 'wavelet_energy': Wavelet energy (good for gradual changes)
            - 'tau': Original WHT sparsity (good for structural changes)
            Default: all methods
    
    Returns:
        method_results: Dict with results from each method
        consensus_seams: Seams detected by multiple methods
    """
    if methods is None:
        methods = ['period', 'wavelet', 'tau']
    
    results = {}
    all_seams = []
    
    # Period-based detection
    if 'period' in methods:
        try:
            pos, changes, seams = detect_seams_period(signal, window_size, stride)
            results['period'] = {
                'positions': pos,
                'values': changes,
                'seams': seams
            }
            all_seams.extend(seams)
        except Exception as e:
            results['period'] = {'error': str(e)}
    
    # Wavelet detail detection
    if 'wavelet' in methods and PYWT_AVAILABLE:
        try:
            pos, detail, seams = detect_seams_wavelet(signal)
            results['wavelet'] = {
                'positions': pos,
                'values': detail,
                'seams': seams
            }
            all_seams.extend(seams)
        except Exception as e:
            results['wavelet'] = {'error': str(e)}
    
    # Wavelet energy detection
    if 'wavelet_energy' in methods and PYWT_AVAILABLE:
        try:
            pos, energy, seams = detect_seams_wavelet_energy(signal)
            results['wavelet_energy'] = {
                'positions': pos,
                'values': energy,
                'seams': seams
            }
            all_seams.extend(seams)
        except Exception as e:
            results['wavelet_energy'] = {'error': str(e)}
    
    # Original tau-based detection
    if 'tau' in methods:
        from .core import detect_seams as detect_seams_tau
        try:
            pos, tau, seams = detect_seams_tau(signal, window_size, stride)
            results['tau'] = {
                'positions': pos,
                'values': tau,
                'seams': seams
            }
            all_seams.extend(seams)
        except Exception as e:
            results['tau'] = {'error': str(e)}
    
    # Consensus: cluster all seams and keep those detected by 2+ methods
    consensus_seams = _consensus_cluster(all_seams, min_gap=window_size, min_votes=2)
    
    return results, consensus_seams


def auto_select_detector(
    signal: np.ndarray,
    window_size: int = 256
) -> str:
    """
    Automatically select the best detection method based on signal characteristics.
    
    Heuristics:
    - High autocorrelation at lag > 10 → oscillatory → use 'period'
    - High kurtosis in differences → impulsive → use 'wavelet'
    - Otherwise → use 'tau'
    
    Returns:
        Recommended method name: 'period', 'wavelet', or 'tau'
    """
    n = len(signal)
    
    # Check for periodicity
    period = estimate_period_autocorr(signal[:min(n, 1000)], min_period=10)
    is_periodic = period > 20
    
    # Check for impulsiveness (sharp spikes)
    diff = np.diff(signal)
    kurtosis = np.mean(diff**4) / (np.mean(diff**2)**2 + 1e-10)
    is_impulsive = kurtosis > 10
    
    if is_periodic:
        return 'period'
    elif is_impulsive:
        return 'wavelet'
    else:
        return 'tau'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _cluster_nearby(points: List[int], min_gap: int = 50) -> List[int]:
    """Cluster nearby points, returning cluster medians."""
    if len(points) == 0:
        return []
    
    points = sorted(points)
    clusters = [[points[0]]]
    
    for p in points[1:]:
        if p - clusters[-1][-1] < min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    
    return [int(np.median(c)) for c in clusters]


def _consensus_cluster(
    points: List[int],
    min_gap: int = 50,
    min_votes: int = 2
) -> List[int]:
    """
    Cluster points and keep only those with multiple votes.
    
    This implements consensus voting across multiple detectors.
    """
    if len(points) == 0:
        return []
    
    points = sorted(points)
    clusters = [[points[0]]]
    
    for p in points[1:]:
        if p - clusters[-1][-1] < min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    
    # Keep clusters with enough votes
    consensus = []
    for c in clusters:
        if len(c) >= min_votes:
            consensus.append(int(np.median(c)))
    
    return consensus


# =============================================================================
# ENHANCED COMPRESSOR
# =============================================================================

class FlipZipEnhanced:
    """
    Enhanced FlipZip with adaptive regime detection.
    
    Uses the best detection method based on signal characteristics,
    addressing the limitation of WHT-only detection on oscillatory signals.
    """
    
    def __init__(
        self,
        window_size: int = 256,
        quantization_bits: int = 10,
        detection_method: str = 'auto'
    ):
        """
        Args:
            window_size: Window size for transform and detection
            quantization_bits: Bits per quantized coefficient
            detection_method: 'auto', 'period', 'wavelet', 'tau', or 'adaptive'
        """
        self.window_size = window_size
        self.quantization_bits = quantization_bits
        self.detection_method = detection_method
        self.levels = 2 ** quantization_bits
        
        # Import core functions
        from .core import fast_walsh_hadamard, inverse_walsh_hadamard
        self._wht = fast_walsh_hadamard
        self._iwht = inverse_walsh_hadamard
    
    def detect_seams(self, signal: np.ndarray) -> Tuple[List[int], str]:
        """
        Detect seams using configured method.
        
        Returns:
            seams: List of seam indices
            method_used: Name of method actually used
        """
        method = self.detection_method
        
        if method == 'auto':
            method = auto_select_detector(signal, self.window_size)
        
        if method == 'period':
            _, _, seams = detect_seams_period(signal, self.window_size)
        elif method == 'wavelet':
            _, _, seams = detect_seams_wavelet(signal)
        elif method == 'adaptive':
            _, seams = detect_seams_adaptive(signal, self.window_size)
        else:  # tau
            from .core import detect_seams
            _, _, seams = detect_seams(signal, self.window_size)
        
        return seams, method
    
    def analyze(self, signal: np.ndarray) -> dict:
        """
        Analyze signal and return detection results from all methods.
        """
        results = {
            'signal_length': len(signal),
            'auto_selected_method': auto_select_detector(signal, self.window_size),
            'methods': {}
        }
        
        # Run all methods
        try:
            pos, changes, seams = detect_seams_period(signal, self.window_size)
            results['methods']['period'] = {
                'n_seams': len(seams),
                'seams': seams,
                'mean_period_change': float(np.mean(changes))
            }
        except Exception as e:
            results['methods']['period'] = {'error': str(e)}
        
        if PYWT_AVAILABLE:
            try:
                pos, detail, seams = detect_seams_wavelet(signal)
                results['methods']['wavelet'] = {
                    'n_seams': len(seams),
                    'seams': seams
                }
            except Exception as e:
                results['methods']['wavelet'] = {'error': str(e)}
        
        try:
            from .core import detect_seams
            pos, tau, seams = detect_seams(signal, self.window_size)
            results['methods']['tau'] = {
                'n_seams': len(seams),
                'seams': seams,
                'mean_tau': float(np.mean(tau)),
                'std_tau': float(np.std(tau))
            }
        except Exception as e:
            results['methods']['tau'] = {'error': str(e)}
        
        return results

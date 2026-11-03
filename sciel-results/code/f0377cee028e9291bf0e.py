import numpy as np
import time
from scipy.optimize import minimize_scalar

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT.
    Returns array of length 2n-1."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Initialize with a broad Gaussian-like shape in frequency domain
    k = np.arange(n)
    amp = np.exp(-k**2 / (n/4)**2)
    amp[0] = 1.0
    phase = np.zeros(n)
    
    def amp_to_f(amp, phase):
        """Convert spectral params to time-domain f (non-negative)."""
        F = amp * np.exp(1j * phase)
        f_raw = np.fft.ifft(F).real
        f = np.maximum(f_raw, 0.0)
        return f
    
    def compute_from_amp(amp, phase):
        f = amp_to_f(amp, phase)
        score = _compute_score(f)
        return f, score
    
    f, best_score = compute_from_amp(amp, phase)
    best_amp = amp.copy()
    best_phase = phase.copy()
    
    # Exact 1D coordinate descent using scipy.optimize.minimize_scalar
    # This is more reliable than heuristic perturbations
    
    def score_for_amp_i(val, i):
        """Score when only amp[i] is changed to val."""
        test_amp = best_amp.copy()
        test_amp[i] = val
        _, score = compute_from_amp(test_amp, best_phase)
        return score
    
    def score_for_phase_i(val, i):
        """Score when only phase[i] is changed to val."""
        test_phase = best_phase.copy()
        test_phase[i] = val
        _, score = compute_from_amp(best_amp, test_phase)
        return score
    
    rng = np.random.default_rng(42)
    
    # Multiple passes of coordinate descent
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        
        # Amplitude coordinate descent with exact 1D minimization
        for i in range(n):
            if time.time() > deadline - 10.0:
                break
            
            orig_amp = best_amp[i]
            
            # Use minimize_scalar to find optimal amp[i]
            # Search in a reasonable range around current value
            scale = max(abs(orig_amp), 0.1)
            res = minimize_scalar(
                lambda x: score_for_amp_i(x, i),
                bounds=(orig_amp - 5*scale, orig_amp + 5*scale),
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 20}
            )
            
            if res.fun < best_score - 1e-10:
                best_amp[i] = res.x
                best_score = res.fun
                improved = True
        
        # Phase coordinate descent with exact 1D minimization
        for i in range(1, n):  # Skip DC (phase 0)
            if time.time() > deadline - 10.0:
                break
            
            orig_phase = best_phase[i]
            
            # Use minimize_scalar to find optimal phase[i]
            # Search in a reasonable range around current value
            res = minimize_scalar(
                lambda x: score_for_phase_i(x, i),
                bounds=(orig_phase - 3.0, orig_phase + 3.0),
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 20}
            )
            
            if res.fun < best_score - 1e-10:
                best_phase[i] = res.x
                best_score = res.fun
                improved = True
        
        if not improved:
            # Try random perturbations to escape local minima
            escaped = False
            for _ in range(20):
                if time.time() > deadline - 10.0:
                    break
                test_amp = best_amp + rng.normal(0, 0.05, n)
                test_amp = np.maximum(test_amp, 0)
                test_phase = best_phase + rng.normal(0, 0.05, n)
                f_test, score_test = compute_from_amp(test_amp, test_phase)
                if score_test < best_score:
                    best_amp = test_amp
                    best_phase = test_phase
                    best_score = score_test
                    escaped = True
            if not escaped:
                break
    
    f = amp_to_f(best_amp, best_phase)
    return list(f)

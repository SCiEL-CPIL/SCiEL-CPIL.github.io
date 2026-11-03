"""Alternating minimization with fixed flat-top spectral magnitude + phase optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _kaiser_window(N, alpha):
    """Kaiser window of length N with parameter alpha."""
    n = np.arange(N)
    center = (N - 1) / 2.0
    x = (n - center) / center
    if N == 1:
        return np.array([1.0])
    return np.i0(alpha * np.sqrt(1.0 - x**2)) / np.i0(alpha)

def _flat_top_magnitude(alpha):
    """
    Compute the target magnitude spectrum for a flat-top-like window.
    We use a Kaiser-Bessel window's magnitude as the fixed spectral shape.
    Returns magnitude array of length N (one-sided DFT).
    """
    # Create a time-domain window
    w = _kaiser_window(N, alpha)
    # Compute its DFT magnitude
    W = np.fft.rfft(w, n=N)
    mag = np.abs(W)
    # Normalize so that mag[0] = 1 (DC component)
    if mag[0] > 1e-15:
        mag = mag / mag[0]
    return mag

def _soft_project_magnitude(X, target_mag, strength=1.0):
    """
    Soft project the DFT X toward having magnitude target_mag.
    Returns new complex array with magnitudes blended toward target.
    """
    mag_X = np.abs(X)
    phase_X = np.angle(X)
    
    # Blend: new_mag = (1-strength)*mag_X + strength*target_mag
    # But we need to handle the symmetry for real signals
    # X is rfft output: length N//2+1
    new_mag = (1.0 - strength) * mag_X + strength * target_mag
    # Ensure non-negative
    new_mag = np.maximum(new_mag, 1e-15)
    
    return new_mag * np.exp(1j * phase_X)

def _phase_optimize(x_init, target_mag, max_iters=500, lr=0.01):
    """
    Optimize the phase of the DFT to minimize the objective,
    keeping the magnitude close to target_mag.
    
    Uses gradient-free approach: perturb phases, keep improvements.
    """
    best_x = x_init.copy()
    best_val = _objective(best_x)
    
    # Get initial DFT
    X = np.fft.rfft(best_x, n=N)
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Perturb phases
        phase = np.angle(X)
        phase_pert = phase + rng.normal(0, lr, size=len(phase))
        
        # Keep magnitudes close to target
        mag_X = np.abs(X)
        new_mag = (1.0 - 0.1) * mag_X + 0.1 * target_mag
        new_mag = np.maximum(new_mag, 1e-15)
        
        X_new = new_mag * np.exp(1j * phase_pert)
        x_new = np.fft.irfft(X_new, n=N)
        
        val_new = _objective(x_new)
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
            X = X_new
            lr = max(0.001, lr * 0.999)
        else:
            lr = min(0.1, lr * 1.01)
            if it % 100 == 0 and it > 0:
                # Occasional larger perturbation
                phase_big = phase + rng.normal(0, lr * 5, size=len(phase))
                X_big = new_mag * np.exp(1j * phase_big)
                x_big = np.fft.irfft(X_big, n=N)
                val_big = _objective(x_big)
                if val_big < best_val:
                    best_val = val_big
                    best_x = x_big.copy()
                    X = X_big
    
    return best_x, best_val

def _coordinate_descent(x, max_iters=10000):
    """
    Coordinate descent: update one element at a time to minimize the objective.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(456)
    step_size = 0.1
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        idx = rng.integers(0, N)
        delta = rng.normal(0, step_size)
        
        x_trial = best_x.copy()
        x_trial[idx] += delta
        
        val_trial = _objective(x_trial)
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial
            step_size = max(0.001, step_size * 0.999)
        else:
            step_size = min(1.0, step_size * 1.005)
    
    return best_x, best_val

def _peak_targeted_polish(x, max_iters=5000):
    """
    Target the peak of the autoconvolution: identify which lag gives the max |ac|,
    then perturb time-domain samples to reduce that specific peak.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(789)
    step_size = 0.05
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Find the peak lag
        ac = _autoconv_fft(best_x)
        peak_lag = np.argmax(np.abs(ac))
        
        # The peak at lag n involves pairs (j, j+n) for valid j
        # Perturb samples that contribute to this peak
        n_peak = peak_lag
        if n_peak == 0:
            # ac[0] = sum x[j]^2, all samples contribute
            idx = rng.integers(0, N)
        else:
            # ac[n] = sum_j x[j]*x[j+n] for j in [0, N-1-n]
            valid_indices = np.arange(0, N - n_peak)
            if len(valid_indices) == 0:
                idx = rng.integers(0, N)
            else:
                idx = valid_indices[rng.integers(0, len(valid_indices))]
        
        delta = rng.normal(0, step_size)
        x_trial = best_x.copy()
        x_trial[idx] += delta
        
        val_trial = _objective(x_trial)
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial
            step_size = max(0.001, step_size * 0.9995)
        else:
            step_size = min(0.5, step_size * 1.002)
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Try several Kaiser alpha values for the flat-top spectral shape
    alphas = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
    
    for alpha in alphas:
        if _check_deadline():
            break
        
        target_mag = _flat_top_magnitude(alpha)
        
        # Initialize with zero phase (real, symmetric signal)
        X_init = target_mag.astype(complex)
        x_init = np.fft.irfft(X_init, n=N)
        
        # Quick evaluation
        val_init = _objective(x_init)
        if val_init < best_val:
            best_val = val_init
            best_x = x_init.copy()
        
        # Phase optimization
        x_opt, val_opt = _phase_optimize(x_init, target_mag, max_iters=300, lr=0.05)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
        
        if _check_deadline():
            break
    
    if _check_deadline():
        return best_x
    
    # Coordinate descent refinement
    x_cd, val_cd = _coordinate_descent(best_x, max_iters=5000)
    if val_cd < best_val:
        best_val = val_cd
        best_x = x_cd.copy()
    
    if _check_deadline():
        return best_x
    
    # Peak-targeted polish
    x_pol, val_pol = _peak_targeted_polish(best_x, max_iters=5000)
    if val_pol < best_val:
        best_val = val_pol
        best_x = x_pol.copy()
    
    if _check_deadline():
        return best_x
    
    # Final random perturbation refinement
    rng = np.random.default_rng(42)
    step_size = 0.03
    stall_count = 0
    max_iters = 100000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.9998)
        else:
            stall_count += 1
            if stall_count % 10000 == 0:
                step_size = min(0.3, step_size * 1.05)
    
    return best_x
# EVOLVE-BLOCK-END

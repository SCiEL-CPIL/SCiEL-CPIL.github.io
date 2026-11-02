import numpy as np
import time

N = 600
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def _fft_convolve(x):
    """Full linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, size)
    C = np.fft.irfft(X * X, size)
    return C[:2 * n - 1]

def compute_score(x):
    """Normalized maximum of autoconvolution: max(autocorr) / autocorr[0]."""
    C = _fft_convolve(x)
    c0 = C[0]
    if c0 <= 0:
        return np.inf
    return float(np.max(C) / c0)

def _spectral_to_heights(log_amp, phase):
    """
    Reconstruct nonnegative heights from a parametric power spectrum.
    log_amp: shape (N//2+1,) log-magnitudes of the DFT.
    phase:   shape (N//2+1,) phases.
    We take x = IDFT(exp(log_amp) * exp(1j*phase)), then project to nonneg
    by taking abs (which preserves the magnitude spectrum and makes x>=0).
    """
    k = len(log_amp)
    amp = np.exp(log_amp)
    spec = amp * np.exp(1j * phase)
    # Build full spectrum with conjugate symmetry
    full = np.empty(N, dtype=complex)
    full[:k] = spec
    full[k:] = np.conj(spec[1:-1][::-1]) if k > 1 else np.array([0.0])
    # Handle: for even N, rfft length is N//2+1. full[0] and full[N/2] are real.
    # Simpler: use irfft
    x = np.fft.irfft(spec, n=N)
    # Project to nonnegative
    x = np.abs(x)
    return x

def _spectral_params_from_x(x):
    """Extract log-magnitude and phase from a signal."""
    X = np.fft.rfft(x)
    mag = np.abs(X)
    mag = np.maximum(mag, 1e-12)
    log_amp = np.log(mag)
    phase = np.angle(X)
    return log_amp, phase

def _objective(params, smooth_beta=50.0):
    """
    params: (2*(N//2+1),) = [log_amp (N//2+1), phase (N//2+1)]
    Returns smooth max of autoconvolution / autoconv[0].
    """
    k = N // 2 + 1
    log_amp = params[:k]
    phase = params[k:]
    x = _spectral_to_heights(log_amp, phase)
    C = _fft_convolve(x)
    c0 = C[0]
    if c0 < 1e-10:
        return 1e10
    # Smooth max: (1/beta) * log(sum(exp(beta * C)))
    C_scaled = beta_safe = C / c0
    m = np.max(C_scaled)
    smooth_max = m + np.log(np.sum(np.exp(beta_safe * (C_scaled - m)))) / smooth_beta
    return smooth_max

def _objective_grad(params, smooth_beta=50.0):
    """Numerical gradient via central differences (sparse, few params)."""
    k = N // 2 + 1
    n_params = 2 * k
    grad = np.zeros(n_params)
    h = 1e-5
    f0 = _objective(params, smooth_beta)
    for i in range(n_params):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h
        p_minus[i] -= h
        grad[i] = (_objective(p_plus, smooth_beta) - _objective(p_minus, smooth_beta)) / (2 * h)
    return grad

def _optimize_spectral(log_amp0, phase0, time_budget):
    """
    Optimize the spectral parameters to minimize the smooth max of autoconvolution.
    Uses a simple gradient descent with adaptive step size.
    """
    k = N // 2 + 1
    n_params = 2 * k
    params = np.concatenate([log_amp0, phase0])
    
    # Normalize: fix the mean log_amp so that sum(x) is preserved
    # Actually, let's work with the normalized spectrum
    
    best_params = params.copy()
    best_score = compute_score(_spectral_to_heights(log_amp0, phase0))
    
    t_start = time.time()
    lr = 0.01
    smooth_beta = 30.0
    
    # Pre-compute gradient
    try:
        grad = _objective_grad(params, smooth_beta)
    except Exception:
        return best_params, best_score
    
    # Gradient descent with line search
    max_iter = 200
    for it in range(max_iter):
        if time.time() - t_start > time_budget:
            break
        
        # Gradient step
        params_new = params - lr * grad
        
        # Evaluate
        try:
            f_new = _objective(params_new, smooth_beta)
        except Exception:
            lr *= 0.5
            continue
        
        x_new = _spectral_to_heights(params_new[:k], params_new[k:])
        C = _fft_convolve(x_new)
        c0 = C[0]
        if c0 < 1e-10:
            lr *= 0.5
            continue
        score_new = float(np.max(C) / c0)
        
        if score_new < best_score:
            best_params = params_new.copy()
            best_score = score_new
            params = params_new.copy()
            # Recompute gradient
            try:
                grad = _objective_grad(params, smooth_beta)
            except Exception:
                break
            lr = min(lr * 1.1, 0.1)
        else:
            lr *= 0.7
        
        # Increase smoothness as we converge
        if it % 20 == 0 and it > 0:
            smooth_beta = min(smooth_beta * 1.5, 200.0)
    
    return best_params, best_score

def construct():
    t0 = time.time()
    remaining = DEADLINE - SAFETY_MARGIN - (time.time() - t0)
    
    # Start from the seed solution
    eps = (np.sqrt(N) - 1.0) / (N - 1.0)
    x_seed = np.full(N, eps)
    x_seed[0] = 1.0
    
    best_x = x_seed.copy()
    best_score = compute_score(best_x)
    
    # Extract spectral parameters from seed
    log_amp0, phase0 = _spectral_params_from_x(x_seed)
    
    # Optimize in spectral domain
    time_budget = max(remaining - 10.0, 5.0)
    try:
        best_params, opt_score = _optimize_spectral(log_amp0, phase0, time_budget)
        if opt_score < best_score:
            best_x = _spectral_to_heights(best_params[:N//2+1], best_params[N//2+1:])
            best_score = opt_score
    except Exception:
        pass
    
    # Final verification
    if time.time() - t0 < DEADLINE - SAFETY_MARGIN:
        final_score = compute_score(best_x)
        if final_score < best_score:
            best_score = final_score
    
    return best_x.tolist()
# EVOLVE-BLOCK-END

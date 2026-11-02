import numpy as np
import time
from scipy.optimize import minimize, fmin_l_bfgs_b
import warnings

warnings.filterwarnings('ignore')

N = 400
START_TIME = time.time()
TIME_LIMIT = 140.0  # 10s safety margin

def _check_deadline():
    return time.time() - START_TIME > TIME_LIMIT

def _autoconv_fft(x):
    """Compute autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute the objective: max |autoconv| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    m = np.max(np.abs(ac))
    return m / (s * s)

def _objective_and_grad(x):
    """
    Compute objective and gradient using spectral domain.
    
    The objective is max_k |C[k]| / S^2 where C = x * x (autoconv) and S = sum(x).
    
    In frequency domain: C_hat = |X|^2, so C[k] = ifft(|X|^2)[k].
    
    We use the soft-max trick: for the peak, we can compute gradient via FFT.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    X = np.fft.rfft(x, n=L_fft)
    X_conj = np.conj(X)
    C_hat = X * X_conj  # |X|^2, real and non-negative
    C = np.fft.irfft(C_hat, n=L_fft)[:L]
    
    abs_C = np.abs(C)
    m = np.max(abs_C)
    
    if m < 1e-15:
        return m / (s * s), np.zeros_like(x)
    
    # Objective: f = m / s^2
    # We need df/dx. The peak m is at some index k*.
    # m = |C[k*]|. If C[k*] > 0, m = C[k*], else m = -C[k*].
    
    sign_k = np.sign(C[np.argmax(abs_C)])
    if sign_k == 0:
        sign_k = 1.0
    
    # C[k] = ifft(|X|^2)[k] = (1/L_fft) * sum_j |X[j]|^2 * exp(2*pi*i*j*k/L_fft)
    # dC[k]/dx = ?
    # X[j] = sum_n x[n] * exp(-2*pi*i*j*n/L_fft)
    # |X[j]|^2 = X[j] * conj(X[j])
    # d|X[j]|^2 / dx[n] = conj(X[j]) * exp(-2*pi*i*j*n/L_fft) + X[j] * exp(2*pi*i*j*n/L_fft)
    #                     = 2 * Re[conj(X[j]) * exp(-2*pi*i*j*n/L_fft)]
    
    # C[k] = (1/L_fft) * sum_j |X[j]|^2 * exp(2*pi*i*j*k/L_fft)
    # dC[k]/dx[n] = (1/L_fft) * sum_j 2*Re[conj(X[j]) * exp(-2*pi*i*j*n/L_fft)] * exp(2*pi*i*j*k/L_fft)
    
    # This is getting complex. Let's use a simpler approach:
    # dC[k]/dx = 2 * x * (delta_k shifted) -- no, that's for linear.
    
    # Actually, for autoconvolution C = x * x:
    # C[k] = sum_n x[n] * x[k-n]
    # dC[k]/dx[m] = x[k-m] + x[m-k]  (for valid range)
    
    # So the gradient of C[k] w.r.t. x is: x[k-m] + x[m-k] for m in range.
    # This is essentially 2 * (x reversed) convolved with delta at k... 
    # Actually: dC[k]/dx[m] = x[k-m] + x[m-k]
    
    # Let's compute this directly.
    k_star = np.argmax(abs_C)
    grad_C_k = np.zeros(N)
    for m in range(N):
        # x[k_star - m] if in range
        idx1 = k_star - m
        if 0 <= idx1 < N:
            grad_C_k[m] += x[idx1]
        idx2 = m - k_star
        if 0 <= idx2 < N:
            grad_C_k[m] += x[idx2]
    
    # Objective: f = sign_k * C[k_star] / s^2
    # df/dx[m] = sign_k * grad_C_k[m] / s^2 - 2 * sign_k * C[k_star] / s^3 * 1
    #          = sign_k * (grad_C_k[m] / s^2 - 2 * C[k_star] / s^3)
    
    val = sign_k * C[k_star] / (s * s)
    grad = sign_k * (grad_C_k / (s * s) - 2.0 * C[k_star] / (s * s * s))
    
    return val, grad

def _spectral_init(n_points=N, seed=42):
    """
    Initialize using spectral phase retrieval idea.
    Start with random phases in frequency domain, take IFFT to get time-domain signal.
    """
    rng = np.random.RandomState(seed)
    L_fft = 1
    while L_fft < 2 * N - 1:
        L_fft *= 2
    
    # Flat magnitude spectrum
    mag = np.ones(L_fft // 2 + 1)
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    X = mag * np.exp(1j * phases)
    X[0] = mag[0]  # DC component real
    
    x = np.fft.irfft(X, n=L_fft)[:N]
    
    # Normalize
    s = np.sum(x)
    if abs(s) < 1e-10:
        x = np.ones(N)
    else:
        x = x / s * N  # Make sum = N
    
    return x

def _amplitude_optimize(x0, max_iter=5000, tol=1e-10):
    """
    Optimize in the time domain using L-BFGS-B.
    """
    best_x = x0.copy()
    best_f = _objective_only(x0)
    
    def fun_and_grad(x):
        if _check_deadline():
            return best_f, np.zeros_like(x)
        f, g = _objective_and_grad(x)
        return f, g
    
    try:
        result = minimize(
            fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 10000}
        )
        
        if result.success or result.fun < best_f:
            best_x = result.x
            best_f = result.fun
    except Exception:
        pass
    
    return best_x, best_f

def _peak_targeted_polish(x, max_iter=10000):
    """
    Coordinate descent targeting the peak.
    For each coordinate, try to reduce the peak.
    """
    best_x = x.copy()
    best_f = _objective_only(x)
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Get current autoconv
    X = np.fft.rfft(best_x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    abs_C = np.abs(C)
    k_star = np.argmax(abs_C)
    
    # Precompute gradient direction for the peak
    grad_C_k = np.zeros(N)
    for m in range(N):
        idx1 = k_star - m
        if 0 <= idx1 < N:
            grad_C_k[m] += best_x[idx1]
        idx2 = m - k_star
        if 0 <= idx2 < N:
            grad_C_k[m] += best_x[idx2]
    
    s = np.sum(best_x)
    sign_k = np.sign(C[k_star])
    if sign_k == 0:
        sign_k = 1.0
    
    # Gradient of objective
    grad_obj = sign_k * (grad_C_k / (s * s) - 2.0 * C[k_star] / (s * s * s))
    
    # Line search along -grad_obj
    step = 0.1
    for _ in range(max_iter):
        if _check_deadline():
            break
        
        x_new = best_x - step * grad_obj
        f_new = _objective_only(x_new)
        
        if f_new < best_f:
            best_x = x_new
            best_f = f_new
            # Recompute gradient
            X = np.fft.rfft(best_x, n=L_fft)
            C = np.fft.irfft(X * X, n=L_fft)[:L]
            abs_C = np.abs(C)
            k_star = np.argmax(abs_C)
            grad_C_k = np.zeros(N)
            for m in range(N):
                idx1 = k_star - m
                if 0 <= idx1 < N:
                    grad_C_k[m] += best_x[idx1]
                idx2 = m - k_star
                if 0 <= idx2 < N:
                    grad_C_k[m] += best_x[idx2]
            s = np.sum(best_x)
            sign_k = np.sign(C[k_star])
            if sign_k == 0:
                sign_k = 1.0
            grad_obj = sign_k * (grad_C_k / (s * s) - 2.0 * C[k_star] / (s * s * s))
            step = min(step * 1.1, 1.0)
        else:
            step *= 0.5
            if step < 1e-6:
                break
    
    return best_x, best_f

def _random_restart_optimize(seed, max_lbgfs_iter=2000):
    """
    Single restart: spectral init + L-BFGS + polish.
    """
    if _check_deadline():
        return None, None
    
    x = _spectral_init(seed=seed)
    
    # L-BFGS optimization
    x, f = _amplitude_optimize(x, max_iter=max_lbgfs_iter)
    
    if _check_deadline():
        return x, f
    
    # Peak-targeted polish
    x, f = _peak_targeted_polish(x, max_iter=5000)
    
    return x, f

def construct():
    """
    Main entry point.
    Returns a sequence of signed step heights.
    """
    # Start with constant sequence as baseline
    x_best = np.ones(N)
    f_best = _objective_only(x_best)  # Should be 1/N = 0.0025
    
    # Try multiple random restarts
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999]
    
    for seed in seeds:
        if _check_deadline():
            break
        
        x_candidate, f_candidate = _random_restart_optimize(seed)
        
        if x_candidate is not None and f_candidate is not None:
            if f_candidate < f_best:
                x_best = x_candidate
                f_best = f_candidate
    
    # Final check: ensure the result is valid
    s = np.sum(x_best)
    if abs(s) < 1e-10:
        return np.ones(N)
    
    return x_best

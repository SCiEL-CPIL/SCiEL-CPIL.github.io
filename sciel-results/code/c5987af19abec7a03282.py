import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

def _hermite_vals(x):
    x = np.asarray(x, dtype=float)
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    return H0, H4, H8, H12

def _g_and_fft(coeffs, x):
    a0, a4, a8 = coeffs
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    H0, H4, H8, H12 = _hermite_vals(x)
    gauss = np.exp(-x**2 / 2.0)
    
    g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss
    
    # F[H_n(x)e^{-x^2/2}](k) = sqrt(2pi) * i^n * H_n(k) * e^{-k^2/2}
    # For even n, i^n = 1
    g_hat = np.sqrt(2.0*np.pi) * (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss
    
    return g, g_hat

def _compute_A(f_vals, x):
    abs_f = np.abs(f_vals)
    norm2 = np.sqrt(np.trapz(abs_f**2, x))
    norm1 = np.trapz(abs_f, x)
    if norm1 < 1e-15:
        return np.inf
    return norm2 / norm1

def _score(coeffs):
    if time.time() > _DEADLINE:
        return None
    x = np.linspace(-12, 12, 20000)
    g, g_hat = _g_and_fft(coeffs, x)
    A_g = _compute_A(g, x)
    A_g_hat = _compute_A(g_hat, x)
    if not np.isfinite(A_g) or not np.isfinite(A_g_hat):
        return np.inf
    return A_g * A_g_hat

def construct():
    best_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_score = _score(best_coeffs)
    if best_score is None:
        best_score = np.inf
    
    rng = np.random.default_rng(123)
    
    # Local search with multiple starting points
    for trial in range(30):
        if time.time() > _DEADLINE - 20:
            break
        if trial == 0:
            start = best_coeffs.copy()
        else:
            start = best_coeffs + rng.normal(0, 0.05, size=3)
        
        try:
            res = minimize(
                lambda c: _score(c) if _score(c) is not None else np.inf,
                start,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    if time.time() > _DEADLINE:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    return np.array(best_coeffs)

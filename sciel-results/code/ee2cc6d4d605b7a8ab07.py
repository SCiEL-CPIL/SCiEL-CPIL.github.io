import numpy as np
from scipy.signal import windows
from scipy.optimize import minimize
import time

N = 400

def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes."""
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def _soft_max_sidelobe(f, lam=10.0):
    """Compute a smooth approximation of the max sidelobe."""
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    sidelobes = np.delete(ac, center)
    abs_sidelobes = np.abs(sidelobes)
    # Soft max
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-10:
        return 0.0
    # Shift to avoid overflow
    shifted = abs_sidelobes - max_val
    return max_val + (1.0/lam) * np.log(np.sum(np.exp(lam * shifted)))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    x = np.linspace(-0.5, 0.5, N)
    
    # Parametric model: Sum of 3 symmetric Gaussians
    # f(x) = a1 * exp(-b1 * x^2) + a2 * exp(-b2 * x^2) + a3 * exp(-b3 * x^2)
    # Parameters: [a1, b1, a2, b2, a3, b3]
    # Constraints: b_i > 0
    
    def objective(params):
        a1, b1, a2, b2, a3, b3 = params
        if b1 <= 0 or b2 <= 0 or b3 <= 0:
            return 1e10
        f = a1 * np.exp(-b1 * x**2) + a2 * np.exp(-b2 * x**2) + a3 * np.exp(-b3 * x**2)
        norm = np.linalg.norm(f)
        if norm < 1e-10:
            return 1e10
        f = f / norm
        return _soft_max_sidelobe(f, lam=20.0)

    # Initial guesses
    # Try a few different starting points
    best_val = float('inf')
    best_f = None
    
    # Starting point 1: Single Gaussian
    init1 = [1.0, 10.0, 0.0, 10.0, 0.0, 10.0]
    # Starting point 2: Two Gaussians
    init2 = [1.0, 5.0, 0.5, 20.0, 0.0, 10.0]
    # Starting point 3: Three Gaussians
    init3 = [1.0, 3.0, 0.5, 10.0, 0.2, 30.0]
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.success and res.fun < best_val:
                best_val = res.fun
                params = res.x
                a1, b1, a2, b2, a3, b3 = params
                f = a1 * np.exp(-b1 * x**2) + a2 * np.exp(-b2 * x**2) + a3 * np.exp(-b3 * x**2)
                norm = np.linalg.norm(f)
                if norm > 1e-10:
                    best_f = f / norm
        except Exception:
            pass

    # If optimization failed or didn't improve, fallback to Kaiser
    if best_f is None:
        # Try a quick grid search on Kaiser beta
        best_kaiser_val = float('inf')
        best_kaiser_f = None
        for beta in np.arange(2, 20, 1):
            if time.time() > deadline - 5:
                break
            f = windows.kaiser(N, beta)
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            f = f / norm
            val = _get_max_sidelobe(f)
            if val < best_kaiser_val:
                best_kaiser_val = val
                best_kaiser_f = f.copy()
        
        if best_kaiser_f is not None:
            return best_kaiser_f
        else:
            f = windows.kaiser(N, 10)
            return f / np.linalg.norm(f)
            
    return best_f

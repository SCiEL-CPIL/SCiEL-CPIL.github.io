"""
Minimize the autocorrelation inequality value using FFT-accelerated gradient
descent with a log-sum-exp smooth surrogate for the max operator, combined
with a low-dimensional Fourier parameterization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s limit


def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))


def _fft_autocorr(h):
    """Compute full autocorrelation via FFT (zero-padded)."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size=size)
    ac = np.fft.irfft(H * np.conj(H), size=size)
    # ac[0..n-1] is the linear autocorrelation: sum h[i]h[i+k]
    return ac[:n]


def _soft_max(c, beta=50.0):
    """Log-sum-exp approximation of max(c)."""
    c_max = np.max(c)
    return c_max + np.log(np.sum(np.exp((c - c_max) * beta)) / len(c)) / beta


def _make_h_fourier(coeffs, M):
    """
    Construct h as a Fourier series:
    h[n] = a0 + sum_{j=1}^{M} (a_j cos(2*pi*j*n/N) + b_j sin(2*pi*j*n/N))
    coeffs = [a0, a1, b1, a2, b2, ..., aM, bM]
    """
    n = np.arange(N)
    h = np.full(N, coeffs[0])
    for j in range(1, M + 1):
        a = coeffs[2 * (j - 1)]
        b = coeffs[2 * (j - 1) + 1]
        h += a * np.cos(2 * np.pi * j * n / N) + b * np.sin(2 * np.pi * j * n / N)
    return h


def _fourier_grad(coeffs, M, beta=50.0):
    """
    Compute gradient of the smooth objective w.r.t. Fourier coefficients.
    Uses FFT for fast autocorrelation and analytical gradient of log-sum-exp.
    """
    h = _make_h_fourier(coeffs, M)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e12, np.zeros_like(coeffs)
    
    # Compute autocorrelation via FFT
    ac = _fft_autocorr(h)
    
    # Soft max of autocorrelation
    c_max = np.max(ac)
    exp_vals = np.exp((ac - c_max) * beta)
    sum_exp = np.sum(exp_vals)
    soft_max_val = c_max + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. ac elements
    weights = exp_vals / sum_exp  # softmax weights
    
    # d(soft_max)/d(ac[k]) = weights[k]
    # ac[k] = sum_i h[i] * h[i+k] (for k=0..N-1, where h is zero-padded)
    # d(ac[k])/d(h[m]) = h[m+k] + h[m-k] (with appropriate boundary handling)
    # For full convolution: ac_full[k] = sum_i h[i] h[i+k]
    # d(ac_full[k])/d(h[m]) = h[m+k] if m+k < N else 0 + h[m-k] if m-k >= 0 else 0
    
    # Build gradient of soft_max w.r.t. h
    grad_h = np.zeros(N)
    for k in range(N):
        w = weights[k]
        if w < 1e-15:
            continue
        # d(ac[k])/d(h[m]) for m=0..N-1
        # ac[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k]
        # d/dh[m] = h[m+k] (if m+k < N) + h[m-k] (if m-k >= 0)
        idx_plus = np.arange(N - k)  # m where m+k < N
        idx_minus = np.arange(k, N)  # m where m-k >= 0
        grad_h[idx_plus] += w * h[idx_plus + k]
        grad_h[idx_minus] += w * h[idx_minus - k]
    
    # Objective: V = 2*N*soft_max / s^2
    # dV/d(h) = 2*N * (d(soft_max)/dh * s^2 - soft_max * 2*s * ds/dh) / s^4
    #         = 2*N * (grad_h * s^2 - soft_max * 2*s * ones) / s^4
    #         = 2*N * grad_h / s^2 - 4*N * soft_max / s^3
    obj = abs(2.0 * N * soft_max_val / (s * s))
    grad_obj_h = 2.0 * N * grad_h / (s * s) - 4.0 * N * soft_max_val / (s ** 3)
    
    # Chain rule: d(obj)/d(coeffs[j]) = sum_m grad_obj_h[m] * d(h[m])/d(coeffs[j])
    grad_coeffs = np.zeros_like(coeffs)
    grad_coeffs[0] = np.sum(grad_obj_h)  # d(h)/d(a0) = 1
    
    n = np.arange(N)
    for j in range(1, M + 1):
        a_idx = 2 * (j - 1)
        b_idx = 2 * (j - 1) + 1
        cos_j = np.cos(2 * np.pi * j * n / N)
        sin_j = np.sin(2 * np.pi * j * n / N)
        grad_coeffs[a_idx] = np.sum(grad_obj_h * cos_j)
        grad_coeffs[b_idx] = np.sum(grad_obj_h * sin_j)
    
    return obj, grad_coeffs


def _fourier_obj(coeffs, M, beta=50.0):
    """Objective function for Fourier coefficient optimization."""
    obj, _ = _fourier_grad(coeffs, M, beta)
    return obj


def construct():
    """Construct step heights minimizing the objective value."""
    start = time.time()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def _update(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Try different numbers of Fourier modes
    for M in [3, 5, 8, 12]:
        if time.time() - start > _DEADLINE - 10:
            break
        
        dim = 2 * M + 1  # a0 + M pairs
        
        # Multiple random starts
        n_starts = max(3, int((_DEADLINE - (time.time() - start)) / 20))
        n_starts = min(n_starts, 10)
        
        for trial in range(n_starts):
            if time.time() - start > _DEADLINE - 5:
                break
            
            # Random initial coefficients
            np.random.seed(42 + trial * 7 + M)
            coeffs0 = np.zeros(dim)
            coeffs0[0] = 1.0  # constant term
            coeffs0[1:] = np.random.randn(dim - 1) * 0.3
            
            # L-BFGS-B optimization
            try:
                res = minimize(
                    _fourier_obj, coeffs0, args=(M, 50.0),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.x is not None:
                    h = _make_h_fourier(res.x, M)
                    _update(h)
                    
                    # Try with different beta values for polishing
                    for beta in [100.0, 200.0]:
                        if time.time() - start > _DEADLINE - 5:
                            break
                        try:
                            res2 = minimize(
                                _fourier_obj, res.x, args=(M, beta),
                                method='L-BFGS-B',
                                jac=True,
                                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                            )
                            if res2.x is not None:
                                h2 = _make_h_fourier(res2.x, M)
                                _update(h2)
                        except Exception:
                            pass
            except Exception:
                pass
    
    # Final polish: use Nelder-Mead on the best solution found
    if time.time() - start < _DEADLINE - 3:
        # Convert best_h back to Fourier coefficients via least squares
        n = np.arange(N)
        # Build Fourier basis matrix
        M_max = 15
        A = np.zeros((N, 2 * M_max + 1))
        A[:, 0] = 1.0
        for j in range(1, M_max + 1):
            A[:, 2 * (j - 1)] = np.cos(2 * np.pi * j * n / N)
            A[:, 2 * (j - 1) + 1] = np.sin(2 * np.pi * j * n / N)
        
        try:
            coeffs_init, _, _, _ = np.linalg.lstsq(A, best_h, rcond=None)
            
            def _nm_obj(c):
                h = _make_h_fourier(c[:2 * M_max + 1], M_max)
                return _value(h)
            
            res_nm = minimize(
                _nm_obj, coeffs_init, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.x is not None:
                h_nm = _make_h_fourier(res_nm.x[:2 * M_max + 1], M_max)
                _update(h_nm)
        except Exception:
            pass
    
    return best_h

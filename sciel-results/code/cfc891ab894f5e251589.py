"""Optimized second autocorrelation inequality construction."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    For nonnegative h on equal intervals. The autoconvolution is discrete correlation.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    # Normalize to L1 = 1 for scale invariance
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    # Discrete autocorrelation (full mode)
    conv = np.correlate(h_norm, h_norm, mode='full')
    # Supremum norm = max absolute value; for nonnegative h, all values >= 0
    conv_max = np.max(np.abs(conv))
    if conv_max <= 0:
        return 0.0
    return l2_sq / conv_max

def _score_complex(g):
    """Score for complex parameterization g where h = |g|^2.
    This removes non-negativity constraints.
    """
    h = np.abs(g) ** 2
    return _score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_h = np.ones(N) / N  # Uniform fallback
    
    # Initial best: uniform
    best_score = _score(best_h)
    
    # Try several initial conditions with complex parameterization
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Different initial conditions
        if trial == 0:
            # Uniform complex
            g = np.ones(N, dtype=np.complex128) / np.sqrt(N)
        elif trial == 1:
            # Peak at center
            g = np.zeros(N, dtype=np.complex128)
            g[N // 2] = 1.0
        elif trial == 2:
            # Gaussian-like
            x = np.linspace(-1, 1, N)
            g = np.exp(-2.0 * x ** 2).astype(np.complex128)
        elif trial == 3:
            # Random
            g = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex128)
            g /= np.sqrt(np.sum(np.abs(g) ** 2))
        elif trial == 4:
            # Bimodal
            g = np.zeros(N, dtype=np.complex128)
            g[N // 4] = 1.0
            g[3 * N // 4] = 1.0
        elif trial == 5:
            # Cosine
            x = np.linspace(0, np.pi, N)
            g = (np.cos(x) + 0j).astype(np.complex128)
        elif trial == 6:
            # Sinc-like
            x = np.linspace(-4, 4, N)
            with np.errstate(divide='ignore', invalid='ignore'):
                g = (np.sinc(x / np.pi)).astype(np.complex128)
            g[np.isnan(g)] = 0
        elif trial == 7:
            # Triangle
            g = np.maximum(0, 1 - np.abs(np.arange(N) - N / 2) / (N / 2)).astype(np.complex128)
        else:
            # Random with varying sparsity
            sparsity = 0.5 + 0.5 * (trial - 8) / 12.0
            mask = np.random.rand(N) < sparsity
            g = (np.random.randn(N) + 1j * np.random.randn(N)) * mask
            g = g.astype(np.complex128)
            norm = np.sqrt(np.sum(np.abs(g) ** 2))
            if norm > 0:
                g /= norm
    
        def objective(g_flat):
            g = g_flat[:N * 2].reshape(N, 2)
            g_complex = g[:, 0] + 1j * g[:, 1]
            s = _score_complex(g_complex)
            return -s
        
        def jac(g_flat):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros(2 * N)
            for i in range(2 * N):
                g_plus = g_flat.copy()
                g_plus[i] += eps
                g_minus = g_flat.copy()
                g_minus[i] -= eps
                grad[i] = (objective(g_plus) - objective(g_minus)) / (2 * eps)
            return grad
        
        # Flatten initial g
        g_init = np.zeros(2 * N)
        g_init[0::2] = g.real
        g_init[1::2] = g.imag
        
        try:
            res = minimize(
                objective,
                g_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000}
            )
            g_opt_flat = res.x
            g_opt = (g_opt_flat[0::2] + 1j * g_opt_flat[1::2]).reshape(N)
            h_opt = np.abs(g_opt) ** 2
            
            # Normalize
            l1 = np.sum(h_opt)
            if l1 > 0:
                h_opt = h_opt / l1
            else:
                h_opt = best_h
            
            s = _score(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Polish pass with tighter tolerance if time permits
    if time.time() < deadline - 20:
        try:
            g_best = np.sqrt(np.maximum(best_h, 0)) + 0j
            g_init = np.zeros(2 * N)
            g_init[0::2] = g_best.real
            g_init[1::2] = g_best.imag
            
            def objective_polish(g_flat):
                g = g_flat[:N * 2].reshape(N, 2)
                g_complex = g[:, 0] + 1j * g[:, 1]
                return -_score_complex(g_complex)
            
            res = minimize(
                objective_polish,
                g_init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            g_opt_flat = res.x
            g_opt = (g_opt_flat[0::2] + 1j * g_opt_flat[1::2]).reshape(N)
            h_opt = np.abs(g_opt) ** 2
            l1 = np.sum(h_opt)
            if l1 > 0:
                h_opt = h_opt / l1
            else:
                h_opt = best_h
            
            s = _score(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    return list(best_h)

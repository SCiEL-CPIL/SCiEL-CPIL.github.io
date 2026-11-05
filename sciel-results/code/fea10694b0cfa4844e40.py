import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve

N = 400
_n = np.arange(N)
_DEADLINE = 140.0

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _make_h_cheb(coeffs, K):
    x = np.linspace(-1, 1, N)
    basis = np.zeros((K, N))
    basis[0] = 1.0
    if K > 1:
        basis[1] = x
    for j in range(2, K):
        basis[j] = 2.0 * x * basis[j-1] - basis[j-2]
    h = basis.T @ coeffs
    return h

def _obj_cheb(coeffs, K):
    h = _make_h_cheb(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    c = np.convolve(h, h)
    max_c = np.max(c)
    val = 2.0 * N * max_c / (s * s)
    return abs(val)

def _make_h_fourier(coeffs, M):
    h = np.full(N, 1.0 / N)
    for j in range(1, M + 1):
        a = coeffs[2 * (j - 1)]
        b = coeffs[2 * (j - 1) + 1]
        h += (a * np.cos(2 * np.pi * j * _n / N) + b * np.sin(2 * np.pi * j * _n / N)) / N
    return h

def _obj_fourier(coeffs, M):
    h = _make_h_fourier(coeffs, M)
    s = h.sum()
    if abs(s) < 1e-12:
        return 1e12
    c = np.convolve(h, h)
    return abs(2.0 * N * c.max() / (s * s))

def _soft_max_gradient_descent(h_init, max_iters=5000, time_limit=40.0):
    h = h_init.copy().astype(float)
    s = np.sum(h)
    if abs(s) < 1e-12:
        h += np.random.normal(0, 1e-6, N)
        s = np.sum(h)
    h = h / s
    
    beta = 10.0
    eta = 0.05
    best_val = _value(h)
    best_h = h.copy()
    t0 = time.time()
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
            
        c = np.convolve(h, h)
        c_max = np.max(c)
        exp_c = np.exp(beta * (c - c_max))
        sum_exp = np.sum(exp_c)
        W = exp_c / sum_exp
        
        # Compute gradient using FFT
        # S2[i] = sum_k h[k] W[i-k]
        conv_h_W = fftconvolve(h, W, mode='full')
        S2 = conv_h_W[:N]
        
        # S1[i] = sum_k h[k-i] W[k]
        W_rev = W[::-1]
        conv_h_Wrev = fftconvolve(h, W_rev, mode='full')
        # Indices 2N-2 down to N-1
        S1 = conv_h_Wrev[2*N-2 : N-2 : -1]
        
        grad = 2.0 * (S1 + S2)
        
        h = h - eta * grad
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += np.random.normal(0, 1e-6, N)
            s = np.sum(h)
        h = h / s
        
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        if it % 100 == 0:
            beta *= 1.05
            eta *= 0.98

    return best_h

def construct():
    start_time = time.time()
    
    best_h = np.ones(N).astype(float)
    best_val = 2.0
    
    def _update(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
            
    # Phase 1: Global search using Chebyshev basis
    for K in [5, 8, 12]:
        if time.time() - start_time > _DEADLINE - 40:
            break
            
        dim = K
        bounds = [(0.5, 5.0)] + [(-3.0, 3.0)] * (K - 1)
        
        remaining = _DEADLINE - 40 - (time.time() - start_time)
        if remaining < 5:
            continue
        maxiter = max(20, int(remaining * 1.2))
        popsize = max(10, min(30, 30 - K))
        
        try:
            res = differential_evolution(
                _obj_cheb, bounds, args=(K,),
                maxiter=maxiter, popsize=popsize,
                tol=0.01, mutation=(0.5, 1.0), recombination=0.7,
                seed=42, polish=False,
                init='sobol' if dim <= 20 else 'latinhypercube'
            )
            if res.x is not None:
                h = _make_h_cheb(res.x, K)
                _update(h)
                
                if time.time() - start_time < _DEADLINE - 20:
                    try:
                        r2 = minimize(_obj_cheb, res.x, args=(K,), method='Nelder-Mead',
                                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                        if r2.x is not None:
                            h2 = _make_h_cheb(r2.x, K)
                            _update(h2)
                    except Exception:
                        pass
        except Exception:
            pass

    # Phase 2: Global search using Fourier basis
    for M in [3, 6, 10]:
        if time.time() - start_time > _DEADLINE - 20:
            break
            
        dim = 2 * M
        bounds = [(-2.0, 2.0)] * dim
        
        remaining = _DEADLINE - 20 - (time.time() - start_time)
        if remaining < 5:
            continue
        maxiter = max(20, int(remaining * 1.0))
        popsize = max(10, min(30, 25 - M))
        
        try:
            res_f = differential_evolution(
                _obj_fourier, bounds, args=(M,),
                maxiter=maxiter, popsize=popsize,
                tol=0.02, mutation=(0.4, 1.0), recombination=0.8,
                seed=123, polish=False,
                init='sobol' if dim <= 20 else 'latinhypercube'
            )
            if res_f.x is not None:
                h_f = _make_h_fourier(res_f.x, M)
                _update(h_f)
        except Exception:
            pass

    # Phase 3: Refine the best candidate using soft-max gradient descent
    if time.time() - start_time < _DEADLINE - 10:
        h_refined = _soft_max_gradient_descent(best_h, max_iters=5000, time_limit=40.0)
        _update(h_refined)
        
        for attempt in range(2):
            if time.time() - start_time > _DEADLINE - 5:
                break
            noise_scale = 0.05 * (attempt + 1)
            h_perturbed = best_h + np.random.normal(0, noise_scale, N)
            h_refined2 = _soft_max_gradient_descent(h_perturbed, max_iters=2000, time_limit=15.0)
            _update(h_refined2)

    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N).astype(float)
        
    return best_h

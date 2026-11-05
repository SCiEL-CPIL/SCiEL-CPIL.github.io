import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_n = np.arange(N)
_DEADLINE = 140.0  # safety margin under 150s limit

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _make_h_cheb(coeffs, K):
    """
    Construct h as linear combination of Chebyshev polynomials.
    h[n] = sum_{j=0}^{K-1} coeffs[j] * T_j(x_n)
    where x_n = linspace(-1, 1, N)
    """
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
    """Objective function for Chebyshev coefficient optimization."""
    h = _make_h_cheb(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    c = np.convolve(h, h)
    max_c = np.max(c)
    val = 2.0 * N * max_c / (s * s)
    return abs(val)

def _make_h_fourier(coeffs, M):
    """
    Band-limited real signal with unit sum:
    h[n] = (1 + sum_{j>=1} a_j cos(2 pi j n/N) + b_j sin(2 pi j n/N)) / N
    """
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

def _soft_max_gradient_descent(h_init, max_iters=2000, time_limit=10.0):
    """
    Refine h using gradient descent on the soft-max of the autoconvolution.
    """
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
    
    # Precompute indices for gradient calculation
    # grad_i = 2 * sum_k W_k * h_{k-i}
    # We can compute this by iterating i and summing over valid k
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
            
        # Compute autoconvolution
        c = np.convolve(h, h)
        
        # Compute softmax weights
        # Shift to avoid overflow
        c_max = np.max(c)
        exp_c = np.exp(beta * (c - c_max))
        sum_exp = np.sum(exp_c)
        W = exp_c / sum_exp
        
        # Compute gradient
        grad = np.zeros(N)
        # grad_i = 2 * sum_{k} W_k * h_{k-i}
        # k ranges from 0 to 2N-2
        # For a fixed i, k-i must be in [0, N-1] => k in [i, i+N-1]
        for i in range(N):
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            if k_start > k_end:
                continue
            # indices in h are k - i, which range from 0 to N-1
            # indices in W are k_start to k_end
            grad[i] = 2.0 * np.dot(W[k_start:k_end], h[:k_end - k_start + 1])
            
        # Update
        h = h - eta * grad
        
        # Normalize
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += np.random.normal(0, 1e-6, N)
            s = np.sum(h)
        h = h / s
        
        # Check improvement
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Schedule
        # Increase beta to sharpen the max approximation
        if it % 50 == 0:
            beta *= 1.1
            eta *= 0.95
            
    return best_h

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
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
                
                # Local polish with Nelder-Mead on coefficients
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
    # This allows exploring the full 400-dimensional space locally
    if time.time() - start_time < _DEADLINE - 10:
        # Try refining the best solution
        h_refined = _soft_max_gradient_descent(best_h, max_iters=3000, time_limit=30.0)
        _update(h_refined)
        
        # Try a few random perturbations and refine
        for attempt in range(2):
            if time.time() - start_time > _DEADLINE - 5:
                break
            noise_scale = 0.05 * (attempt + 1)
            h_perturbed = best_h + np.random.normal(0, noise_scale, N)
            h_refined2 = _soft_max_gradient_descent(h_perturbed, max_iters=2000, time_limit=15.0)
            _update(h_refined2)

    # Final validation
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N).astype(float)
        
    return best_h

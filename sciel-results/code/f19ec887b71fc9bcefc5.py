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

def _smooth_max_obj(h, beta=50.0):
    """
    Smooth surrogate objective using log-sum-exp.
    Minimizes log-sum-exp(conv(h,h)/beta) + penalty for sum(h) != 1.
    This provides smooth gradients unlike hard max.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12, None
    c = np.convolve(h, h)
    # Log-sum-exp of c entries
    max_c = np.max(c)
    lse = max_c + beta * np.log(np.sum(np.exp((c - max_c) / beta)))
    # Penalty for deviation from sum=1
    penalty = 1e6 * (s - 1.0) ** 2
    val = lse / beta + penalty
    return val, None

def _smooth_max_grad(h, beta=50.0):
    """
    Gradient of the smooth log-sum-exp surrogate.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12, np.zeros_like(h)
    c = np.convolve(h, h)
    max_c = np.max(c)
    # Soft weights: w_k = exp((c_k - max_c)/beta) / sum_j exp((c_j - max_c)/beta)
    exp_c = np.exp((c - max_c) / beta)
    sum_exp = np.sum(exp_c)
    w = exp_c / sum_exp
    
    # d/dh_m [sum_k w_k * c_k] where c_k = sum_n h_n h_{k-n}
    # = sum_k w_k * 2 * h_{k-m} for valid indices
    g = np.zeros(N)
    for m in range(N):
        idx = np.arange(N)
        valid = (idx >= 0) & (idx < N)
        # c_k = sum_n h[n]*h[k-n], dc_k/dh_m = 2*h[k-m] if 0<=k-m<N
        # g_m = 2 * sum_k w_k * h[k-m] for valid k-m
        # Equivalently: g = 2 * (w circ conv h) but we need to be careful
        # Let's compute directly
        pass
    
    # More efficient: g_m = 2 * sum_{k=0}^{2N-2} w_k * h_{k-m} * [0<=k-m<N]
    # = 2 * sum_{n=0}^{N-1} h_n * sum_{k=n}^{n+N-1} w_k (where k-m=n means k=m+n)
    # Actually: g_m = 2 * sum_{n: 0<=m+n<2N-1} w_{m+n} * h_n
    # Let's use convolution approach
    # Define extended w: w_ext[i] = w[i] for i in 0..2N-2, else 0
    # g_m = 2 * sum_n h[n] * w[m+n]
    # This is a cross-correlation-like operation
    
    w_ext = np.zeros(2 * N - 1)
    w_ext[:] = w
    # g_m = 2 * sum_n h[n] * w_ext[m+n]
    # For m=0: sum_n h[n] * w_ext[n]
    # For m=1: sum_n h[n] * w_ext[1+n]
    # This is like a sliding dot product
    
    # Use FFT for efficiency
    h_padded = np.zeros(2 * N)
    h_padded[:N] = h
    w_padded = np.zeros(2 * N)
    w_padded[:2*N-1] = w_ext
    
    # g_m = 2 * (h * w_ext)[m] where * is cross-correlation
    # cross_corr[m] = sum_n h[n] * w_ext[m+n]
    # This equals (h reversed) convolved with w_ext, then reversed
    h_rev = h[::-1]
    cross = np.convolve(h_rev, w_ext, mode='full')
    # cross[i] = sum_j h_rev[j] * w_ext[i-j] = sum_j h[N-1-j] * w_ext[i-j]
    # Let k = N-1-j: cross[i] = sum_k h[k] * w_ext[i-(N-1-k)] = sum_k h[k] * w_ext[i-N+1+k]
    # We want g_m = sum_n h[n] * w_ext[m+n]
    # Set i-N+1+k = m+n where n=k: i-N+1 = m, so i = m+N-1
    # g_m = cross[m + N - 1]
    g = 2.0 * cross[N-1 : N-1+N]
    
    # Add penalty gradient
    g += 2.0 * 1e6 * (s - 1.0)
    
    val = (max_c + beta * np.log(sum_exp)) / beta + 1e6 * (s - 1.0) ** 2
    return val, g

def _make_h_cheb(coeffs, K):
    """
    Construct h as linear combination of Chebyshev polynomials.
    h[n] = sum_{j=0}^{K-1} coeffs[j] * T_j(x_n)
    where x_n = linspace(-1, 1, N)
    """
    x = np.linspace(-1, 1, N)
    # Precompute Chebyshev basis
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

def _smooth_grad_optimize(h_init, max_iter=2000, eta=0.01, mu=0.9, beta=50.0):
    """
    Smooth gradient optimization with momentum using log-sum-exp surrogate.
    """
    h = h_init.copy().astype(float)
    s = np.sum(h)
    if abs(s) < 1e-12:
        h += np.random.normal(0, 1e-6, N)
        s = np.sum(h)
    h = h / s
    
    v = np.zeros_like(h)
    best_local_val = np.inf
    best_local_h = h.copy()
    t0 = time.time()
    
    for it in range(max_iter):
        if time.time() - t0 > 15:  # Limit each run to 15s
            break
            
        # Compute smooth objective and gradient
        val_smooth, g = _smooth_max_grad(h, beta)
        
        # Track actual objective
        s_cur = np.sum(h)
        if abs(s_cur) > 1e-12:
            conv = np.convolve(h, h)
            actual_val = 2.0 * N * np.max(conv)
        else:
            actual_val = 1e12
        
        if actual_val < best_local_val:
            best_local_val = actual_val
            best_local_h = h.copy()
        
        # Gradient descent with momentum
        v = mu * v - eta * g
        h = h + v
        
        # Normalize to sum=1
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += np.random.normal(0, 1e-6, N)
            s = np.sum(h)
        h = h / s
        
        # Adaptive step size
        if it % 100 == 0:
            cur_val = 2.0 * N * np.max(np.convolve(h, h))
            if cur_val > best_local_val * 1.5:
                eta *= 0.5
            elif cur_val < best_local_val * 0.95:
                eta = min(eta * 1.1, 0.05)
                
    return best_local_h

def _lbfgs_polish(h_init, max_iter=500, time_limit=20.0):
    """
    L-BFGS-B polish on the full 400-dimensional space using smooth surrogate.
    """
    t0 = time.time()
    
    def _obj_and_grad(x):
        if time.time() - t0 > time_limit:
            return 1e12, np.zeros_like(x)
        val, g = _smooth_max_grad(x, beta=50.0)
        return val, g
    
    try:
        res = minimize(
            _obj_and_grad, h_init,
            jac=True, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8,
                     'maxfun': 1000, 'disp': False}
        )
        if res.success or res.x is not None:
            h_out = res.x
            # Normalize
            s = np.sum(h_out)
            if abs(s) > 1e-12:
                h_out = h_out / s
            return h_out
    except Exception:
        pass
    return h_init

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
    
    # Phase 1: Global search using Chebyshev basis (low-dimensional, smooth)
    # Try different numbers of basis functions
    for K in [3, 5, 7, 10]:
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        dim = K
        # Bounds: T_0 coefficient should be dominant to keep sum(h) nonzero
        bounds = [(0.5, 5.0)] + [(-3.0, 3.0)] * (K - 1)
        
        # Estimate max iterations based on remaining time
        remaining = _DEADLINE - 30 - (time.time() - start_time)
        if remaining < 5:
            continue
        maxiter = max(20, int(remaining * 1.5))
        popsize = max(10, min(40, 35 - K))
        
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
                
                # Local polish with Nelder-Mead
                if time.time() - start_time < _DEADLINE - 15:
                    try:
                        r2 = minimize(_obj_cheb, res.x, args=(K,), method='Nelder-Mead',
                                      options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                        if r2.x is not None:
                            h2 = _make_h_cheb(r2.x, K)
                            _update(h2)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Phase 2: Global search using Fourier basis (alternative smooth family)
    for M in [2, 4, 6, 8]:
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
    
    # Phase 3: Refine using smooth gradient optimization (log-sum-exp surrogate)
    if time.time() - start_time < _DEADLINE - 25:
        # Try smooth gradient descent from best solution
        h_smooth = _smooth_grad_optimize(best_h, max_iter=2000, eta=0.01, mu=0.9, beta=50.0)
        _update(h_smooth)
        
        # Try with different beta values
        for beta_try in [20.0, 100.0]:
            if time.time() - start_time > _DEADLINE - 15:
                break
            h_smooth2 = _smooth_grad_optimize(best_h, max_iter=1000, eta=0.005, mu=0.9, beta=beta_try)
            _update(h_smooth2)
    
    # Phase 4: L-BFGS polish on the best candidate
    if time.time() - start_time < _DEADLINE - 5:
        h_polished = _lbfgs_polish(best_h, max_iter=300, time_limit=25.0)
        _update(h_polished)
        
        # Second polish from slightly perturbed version
        if time.time() - start_time < _DEADLINE - 3:
            h_pert = best_h + np.random.normal(0, 0.005, N)
            h_polished2 = _lbfgs_polish(h_pert, max_iter=200, time_limit=15.0)
            _update(h_polished2)
    
    # Final validation
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N).astype(float)
    
    return best_h

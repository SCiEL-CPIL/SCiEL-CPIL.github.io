import numpy as np
from scipy.optimize import minimize
import time

N = 400
_DEADLINE = 150.0
_START_TIME = None


def _objective_hard(x):
    """Compute the actual evaluation value."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    return abs(2 * N * m / (s * s))


def _soft_max_obj(x, temp=0.05):
    """Differentiable soft-max objective."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    # Numerically stable log-sum-exp
    exp_terms = np.exp((conv - m) / temp)
    logsumexp = m / temp + np.log(np.sum(exp_terms))
    smax = temp * logsumexp
    return smax / (s * s)


def _soft_max_grad(x, temp=0.05):
    """Gradient of soft-max objective using efficient vectorized computation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(x)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    exp_terms = np.exp((conv - m) / temp)
    softmax_weights = exp_terms / np.sum(exp_terms)
    sw = softmax_weights  # length 2N-1
    
    # Compute gradient efficiently using the structure of autoconvolution
    # d(conv[k])/d(h[i]) = h[k-i] (if 0<=k-i<N) + h[i-k] (if 0<=i-k<N)
    # 
    # For each i: g[i] = sum_k sw[k] * (h[k-i] if valid + h[i-k] if valid)
    #
    # First term: sum_{k=i}^{min(i+N-1, 2N-2)} sw[k] * h[k-i]
    # Let j = k-i: sum_{j=0}^{min(N-1, 2N-2-i)} sw[i+j] * h[j]
    #
    # Second term: sum_{k=max(0,i-N+1)}^{i} sw[k] * h[i-k]
    # Let j = i-k: sum_{j=0}^{min(i, N-1)} sw[i-j] * h[j]
    
    # Vectorized computation using slicing
    grad = np.zeros(N)
    
    # First term for all i simultaneously
    # For i in range(N): sum_{j=0}^{min(N-1, 2N-2-i)} sw[i+j] * h[j]
    # This is like a cross-correlation
    
    # Use the fact that for each i, we need sw[i:i+min(N, 2N-1-i)] dotted with h[:min(N, 2N-1-i)]
    # Let's compute this efficiently
    
    # For the first sum: for each i, we want dot(sw[i : i+min(N, 2N-1-i)], h[:min(N, 2N-1-i)])
    # For i=0: dot(sw[0:N], h[0:N])
    # For i=1: dot(sw[1:N+1], h[0:N]) but only if N+1 <= 2N-1, i.e., N>=2
    # Actually for i, the valid range is j from 0 to min(N-1, 2N-2-i)
    # So length = min(N, 2N-1-i)
    
    # Let me just do it with a loop but using np.dot for each i
    for i in range(N):
        # First sum
        max_j = min(N, 2 * N - 1 - i)
        if max_j > 0:
            grad[i] += np.dot(sw[i:i + max_j], h[:max_j])
        
        # Second sum: sum_{j=0}^{min(i, N-1)} sw[i-j] * h[j]
        max_j2 = min(i + 1, N)
        if max_j2 > 0:
            # sw indices: i, i-1, ..., i-(max_j2-1) = i-max_j2+1
            sw_slice = sw[i - max_j2 + 1 : i + 1][::-1]  # reversed
            grad[i] += np.dot(sw_slice, h[:max_j2])
    
    smax = temp * (m / temp + np.log(np.sum(exp_terms)))
    d_obj = grad / (s * s) - 2 * smax / (s * s * s)
    
    return d_obj


def _anneal_temperature(current_time, deadline, total_phases=3):
    """Linearly decrease temperature over time."""
    elapsed = current_time - _START_TIME
    total_time = deadline - _START_TIME
    frac = min(1.0, elapsed / total_time)
    # Temperature from 0.1 to 0.001
    log_temp = np.log(0.1) + frac * (np.log(0.001) - np.log(0.1))
    return np.exp(log_temp)


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_val = 2.0
    best_h = np.ones(N)
    
    deadline = _START_TIME + _DEADLINE - 5.0
    
    # Generate diverse initializations
    inits = []
    
    # Random with different scales
    for i in range(12):
        r = np.random.RandomState(1000 + i * 137)
        inits.append(r.randn(N) * (0.3 + 0.1 * i))
    
    # Linear ramps
    inits.append(np.linspace(-1, 1, N))
    inits.append(np.linspace(1, -1, N))
    
    # Step functions with various splits
    for frac in [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75]:
        h = np.zeros(N)
        h[:int(frac * N)] = 1.0
        h[int(frac * N):] = -0.3
        inits.append(h - np.mean(h))
    
    # Waves with various frequencies
    t = np.linspace(0, 2 * np.pi, N)
    for freq in [1, 2, 3, 5, 8]:
        inits.append(np.cos(freq * t) - np.mean(np.cos(freq * t)))
        inits.append(np.sin(freq * t) - np.mean(np.sin(freq * t)))
    
    # Polynomials
    x = np.linspace(-1, 1, N)
    inits.append(x**2 - np.mean(x**2))
    inits.append(x**3 - np.mean(x**3))
    inits.append(x**4 - np.mean(x**4))
    
    # Comb-like patterns
    for period in [7, 11, 13, 17]:
        h = np.zeros(N)
        h[::period] = 1.0
        h[1::period] = -0.5
        inits.append(h - np.mean(h))
    
    # Optimized initializations with multi-temperature annealing
    for init_idx, init_h in enumerate(inits):
        if time.time() > deadline - 3:
            break
        
        h = init_h.astype(float).copy()
        s = np.sum(h)
        if abs(s) < 1e-12:
            # Add small perturbation
            h += 0.01 * np.random.RandomState(init_idx).randn(N)
            s = np.sum(h)
            if abs(s) < 1e-12:
                continue
        
        # Multi-temperature annealing for this initialization
        temps = [0.1, 0.05, 0.02, 0.01, 0.005]
        current_h = h.copy()
        
        for temp in temps:
            if time.time() > deadline - 3:
                break
            
            try:
                res = minimize(
                    lambda x, t=temp: _soft_max_obj(x, t),
                    current_h,
                    jac=lambda x, t=temp: _soft_max_grad(x, t),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 500}
                )
                
                new_val = _objective_hard(res.x)
                if new_val < best_val:
                    best_val = new_val
                    best_h = res.x.copy()
                
                # Use the result as starting point for next temperature
                if res.fun < _soft_max_obj(current_h, temp):
                    current_h = res.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _soft_max_obj(x, 0.001),
                best_h,
                jac=lambda x: _soft_max_grad(x, 0.001),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            )
            val = _objective_hard(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h

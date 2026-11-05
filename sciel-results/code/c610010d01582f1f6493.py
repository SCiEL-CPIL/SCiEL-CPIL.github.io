"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective_hard(h):
    """Compute the hard objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta):
    """
    Soft-max approximation of the objective.
    Minimizes 2*N*soft_max(conv) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    return val

def _soft_max_grad(h, beta):
    """
    Analytical gradient of the soft-max objective with respect to h.
    Uses efficient convolution-based computation.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # soft_max = m + log(sum_exp)/beta
    # d(soft_max)/d(conv[j]) = exp_shifted[j] / sum_exp
    w = exp_shifted / sum_exp
    
    # Gradient computation using convolution:
    # d(conv[j])/dh[i] = h[i+k] (if valid) + h[i-k] (if valid) where k = j-N+1
    # grad_soft[i] = sum_j w[j] * (h[i+k_j] if 0<=i+k_j<N else 0 + h[i-k_j] if 0<=i-k_j<N else 0)
    
    # We can compute this efficiently:
    # term1[i] = sum_j w[j] * h[i + (j-N+1)] where 0 <= i+(j-N+1) < N
    # term2[i] = sum_j w[j] * h[i - (j-N+1)] where 0 <= i-(j-N+1) < N
    
    # Let's define w_k for k in [-(N-1), N-1] where k = j-N+1, so j = k+N-1
    # w_k = w[k+N-1]
    
    # Create w_k array indexed by offset from -(N-1) to N-1
    w_k = np.zeros(2*N-1)
    for j in range(2*N-1):
        k = j - N + 1
        w_k[k + N - 1] = w[j]
    
    # Now term1[i] = sum_{k: 0<=i+k<N} w_k[i+k] * h[i+k]  -- wait, that's not right
    # term1[i] = sum_j w[j] * h[i + (j-N+1)] where 0 <= i+(j-N+1) < N
    # Let k = j-N+1, then term1[i] = sum_k w_k[k] * h[i+k] where 0 <= i+k < N
    # This is a cross-correlation of w_k with h, but only over valid indices
    
    # Similarly term2[i] = sum_k w_k[k] * h[i-k] where 0 <= i-k < N
    
    # We can compute these using np.correlate or manual convolution
    # Let's use the approach of padding and convolving
    
    # For term1: result1[i] = sum_k w_k[k] * h[i+k] for valid i,k
    # This is equivalent to: for each k, add w_k[k] * h[i+k] to result1[i] for i in [max(0,-k), min(N-1, N-1-k)]
    
    # Let's compute term1 and term2 efficiently
    term1 = np.zeros(N)
    term2 = np.zeros(N)
    
    for k_idx in range(2*N-1):
        k = k_idx - (N-1)
        wk = w_k[k_idx]
        if wk < 1e-15:
            continue
        
        # term1: h[i+k] for 0 <= i+k < N => i in [max(0, -k), min(N-1, N-1-k)]
        i_start = max(0, -k)
        i_end = min(N-1, N-1-k)
        if i_start <= i_end:
            # h[i+k] for i in [i_start, i_end] is h[i_start+k : i_end+k+1]
            h_slice = h[i_start+k : i_end+k+1]
            term1[i_start:i_end+1] += wk * h_slice
        
        # term2: h[i-k] for 0 <= i-k < N => i in [max(0, k), min(N-1, N-1+k)]
        i_start2 = max(0, k)
        i_end2 = min(N-1, N-1+k)
        if i_start2 <= i_end2:
            # h[i-k] for i in [i_start2, i_end2] is h[i_start2-k : i_end2-k+1]
            h_slice2 = h[i_start2-k : i_end2-k+1]
            term2[i_start2:i_end2+1] += wk * h_slice2
    
    grad_soft = term1 + term2
    
    # Now compute the full gradient of the objective:
    # V = 2*N * soft_max / s^2
    # dV/dh[i] = 2*N * (1/s^2) * grad_soft[i] - 2*N * soft_max * 2/s^3 * 1
    #          = 2*N * grad_soft[i] / s^2 - 4*N * soft_max / s^3
    
    soft_max_val = m + np.log(sum_exp) / beta
    grad_full = 2.0 * N * grad_soft / (s * s) - 4.0 * N * soft_max_val / (s * s * s)
    
    return grad_full

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal with positive mean
    for _ in range(15):
        h0 = rng.standard_normal(N)
        h0 += 0.5  # Ensure positive mean
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 2. Sign sequences with DC offset
    for _ in range(15):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.3 * rng.standard_normal()
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 3. Sine waves with various frequencies and phases
    for freq in [1, 2, 3, 5, 7, 10]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.8  # Positive mean
        initializations.append(h0)
        
    # 4. Polynomial perturbations
    for _ in range(10):
        h0 = np.ones(N)
        t = np.linspace(0, 1, N)
        h0 += 0.2 * (rng.random() * t**2 - 0.1)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 5. Gaussian bumps
    for _ in range(10):
        h0 = np.zeros(N)
        center = rng.integers(0, N)
        width = rng.uniform(10, 50)
        t = np.arange(N)
        h0 += np.exp(-0.5 * ((t - center) / width)**2)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)

    # Optimization using scipy's minimize with L-BFGS-B and analytical gradient
    betas = [30.0, 50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
        
        h_curr = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 25.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, h_curr, jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Final polish with high beta and tighter tolerances
    if time.time() < deadline - 15.0:
        for beta in [200.0, 500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 10.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, best_h, jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE_MARGIN = 10  # seconds

def _corr_vec(h):
    """Compute all correlation values."""
    return np.correlate(h, 1 - h, mode='full')

def _value(h):
    """Compute the actual value (lower is better)."""
    corr = _corr_vec(h)
    return np.max(corr) / N * 2

def _fix_sum(h):
    """Project h onto feasible set: sum(h) = N/2, 0 <= h <= 1."""
    diff = N / 2 - np.sum(h)
    h = h + diff / N
    h = np.clip(h, 0, 1)
    # Re-adjust if clipping changed the sum
    diff2 = N / 2 - np.sum(h)
    if abs(diff2) > 1e-10:
        mask = (h > 1e-6) & (h < 1 - 1e-6)
        if np.any(mask):
            h[mask] += diff2 / np.sum(mask)
            h = np.clip(h, 0, 1)
    return h

def _objective_and_grad(h, temperature=1.0):
    """Compute smooth max objective and gradient."""
    corr = _corr_vec(h)
    max_c = np.max(corr)
    shifted = corr - max_c
    exp_shifted = np.exp(shifted / temperature)
    sum_exp = np.sum(exp_shifted)
    
    # Objective: smooth max
    obj = max_c + temperature * np.log(sum_exp)
    
    # Weights for each correlation
    weights = exp_shifted / sum_exp  # shape (2N-1,)
    
    # Gradient computation
    # corr[j] = sum_i h[i+lag] * (1-h[i]) where lag = j - (N-1)
    # d/dh[m] of corr[j] = (1-h[m-lag]) * 1[m-lag valid] - h[m+lag] * 1[m+lag valid]
    grad = np.zeros(N)
    for j in range(2 * N - 1):
        w = weights[j]
        if w < 1e-10:
            continue
        lag = j - (N - 1)
        for m in range(N):
            # Contribution from h[m+lag] term
            if 0 <= m + lag < N:
                grad[m] += w * (1 - h[m - lag]) if (0 <= m - lag < N) else 0
            # Contribution from -h[m+lag] term  
            if 0 <= m + lag < N:
                grad[m] -= w * h[m + lag]
            # Wait, let me redo this more carefully
    
    # Let me recompute the gradient correctly
    # corr[j] = sum_i h[i+lag] * (1-h[i]) where lag = j-(N-1)
    # = sum_i h[i+lag] - sum_i h[i+lag]*h[i]
    
    # d/dh[m] of sum_i h[i+lag]:
    #   = 1 if m-lag is in [0, N-1], i.e., i = m-lag, then h[i+lag] = h[m]
    #   So derivative is 1 if 0 <= m-lag < N
    
    # d/dh[m] of sum_i h[i+lag]*h[i]:
    #   For each i, product h[i+lag]*h[i]
    #   d/dh[m] = h[i] * 1[i+lag=m] + h[i+lag] * 1[i=m]
    #   = h[m-lag] * 1[m-lag valid] + h[m+lag] * 1[m+lag valid]
    
    # So d/dh[m] of corr[j] = 1[0<=m-lag<N] - h[m-lag]*1[0<=m-lag<N] - h[m+lag]*1[0<=m+lag<N]
    #                       = (1-h[m-lag])*1[0<=m-lag<N] - h[m+lag]*1[0<=m+lag<N]
    
    grad = np.zeros(N)
    for j in range(2 * N - 1):
        w = weights[j]
        if w < 1e-10:
            continue
        lag = j - (N - 1)
        for m in range(N):
            contrib = 0.0
            if 0 <= m - lag < N:
                contrib += (1 - h[m - lag])
            if 0 <= m + lag < N:
                contrib -= h[m + lag]
            grad[m] += w * contrib
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    # Temperature schedule
    temperatures = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    
    # Generate diverse initial points
    inits = []
    inits.append(np.full(N, 0.5))
    
    # Sine waves
    for freq in [1, 2, 3, 5, 8, 13]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            t = np.linspace(0, 2 * np.pi * freq, N)
            h = 0.5 + 0.4 * np.sin(t + phase)
            h = _fix_sum(h)
            inits.append(h)
    
    # Random
    for _ in range(30):
        h = np.random.rand(N)
        h = _fix_sum(h)
        inits.append(h)
    
    # Block patterns
    for block_size in [5, 10, 20, 25]:
        for offset in range(0, block_size):
            h = np.zeros(N)
            for i in range(N):
                if (i - offset) % (2 * block_size) < block_size:
                    h[i] = 0.7
                else:
                    h[i] = 0.3
            h = _fix_sum(h)
            inits.append(h)
    
    # Optimize each initial point
    for h0 in inits:
        if time.time() > deadline:
            break
        
        h_curr = h0.copy()
        
        for temp in temperatures:
            if time.time() > deadline:
                break
            
            def obj_func(x):
                obj, _ = _objective_and_grad(x, temperature=temp)
                return obj
            
            def grad_func(x):
                _, grad = _objective_and_grad(x, temperature=temp)
                return grad
            
            try:
                res = minimize(
                    obj_func, h_curr, jac=grad_func,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                h_opt = _fix_sum(res.x)
                val = _value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt
                h_curr = h_opt
            except Exception:
                continue
    
    return best_h

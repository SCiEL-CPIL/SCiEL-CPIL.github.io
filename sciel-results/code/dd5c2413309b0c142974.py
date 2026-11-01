"""Rewritten program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

DEADLINE = 115.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _c3_upper_bound(h):
    """
    Compute the exact C3 upper bound for a given step function h.
    Metric: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    Uses FFT for efficient convolution.
    """
    if np.all(h == 0):
        return float('inf')
    
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    
    valid_C = C[:2*N-1]
    max_abs_conv = np.max(np.abs(valid_C))
    sum_abs_h = np.sum(np.abs(h))
    
    if sum_abs_h < 1e-12:
        return float('inf')
        
    return max_abs_conv / (sum_abs_h ** 2)

def _objective_soft(x):
    """
    Smooth objective function for optimization.
    x is the unconstrained variable, h = tanh(x).
    """
    h = np.tanh(x)
    
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    valid_C = C[:2*N-1]
    
    eps = 1e-10
    abs_C = np.sqrt(valid_C**2 + eps)
    
    T = 5.0
    max_abs_C = np.max(abs_C)
    shifted = (abs_C - max_abs_C) / T
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs_C / T
    soft_max = T * log_sum_exp
    
    abs_h = np.sqrt(h**2 + eps)
    sum_abs_h = np.sum(abs_h)
    
    val = soft_max / (sum_abs_h**2 + 1e-8)
    return val

def _local_search_spin(h, max_iters=10):
    """
    Local search on the discrete spin space {+1, -1}.
    Tries to flip individual spins to improve the exact objective.
    """
    h = h.copy()
    h = np.sign(h)
    if np.any(h == 0):
        h[h == 0] = 1.0
        
    current_val = _c3_upper_bound(h)
    improved = True
    iters = 0
    while improved and iters < max_iters and _time_left() > 5.0:
        improved = False
        iters += 1
        for i in range(N):
            if _time_left() < 5.0:
                return h, current_val
            old_sign = h[i]
            h[i] = -old_sign
            new_val = _c3_upper_bound(h)
            if new_val < current_val - 1e-12:
                current_val = new_val
                improved = True
            else:
                h[i] = old_sign
    return h, current_val

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the upper bound on C3.
    """
    start_time = time.time()
    
    best_h = None
    best_val = float('inf')
    
    # Strategy 1: Gradient-based optimization from multiple initializations
    inits = []
    
    # 1. All +1
    x0_1 = np.full(N, 5.0)
    inits.append(x0_1)
    
    # 2. Alternating
    x0_2 = np.zeros(N)
    x0_2[::2] = 2.0
    x0_2[1::2] = -2.0
    inits.append(x0_2)
    
    # 3. Random
    np.random.seed(42)
    x0_3 = np.random.randn(N) * 2.0
    inits.append(x0_3)
    
    # 4. Blocky
    block_size = 10
    num_blocks = N // block_size
    blocks = np.random.choice([1, -1], size=num_blocks)
    x0_4 = np.repeat(blocks * 2.0, block_size)
    inits.append(x0_4)
    
    # 5. Another random
    np.random.seed(123)
    x0_5 = np.random.randn(N) * 2.0
    inits.append(x0_5)
    
    for i, x0 in enumerate(inits):
        if _time_left() < 10.0:
            break
            
        try:
            res = minimize(_objective_soft, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5})
            
            h_opt = np.tanh(res.x)
            val = _c3_upper_bound(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
    
    # Strategy 2: Discrete local search (Spin glass)
    
    # Discrete init 1: Sign of best continuous
    if best_h is not None:
        h_disc = np.sign(best_h)
        if np.any(h_disc == 0):
            h_disc[h_disc == 0] = 1.0
        h_loc, val_loc = _local_search_spin(h_disc)
        if val_loc < best_val:
            best_val = val_loc
            best_h = h_loc
    
    # Discrete init 2: All +1
    h_disc = np.ones(N)
    h_loc, val_loc = _local_search_spin(h_disc)
    if val_loc < best_val:
        best_val = val_loc
        best_h = h_loc
        
    # Discrete init 3: Alternating
    h_disc = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.float64)
    h_loc, val_loc = _local_search_spin(h_disc)
    if val_loc < best_val:
        best_val = val_loc
        best_h = h_loc
        
    # Discrete init 4: Random
    np.random.seed(42)
    h_disc = np.random.choice([1, -1], size=N).astype(np.float64)
    h_loc, val_loc = _local_search_spin(h_disc)
    if val_loc < best_val:
        best_val = val_loc
        best_h = h_disc
        
    # Discrete init 5: Blocky
    np.random.seed(123)
    block_size = 10
    num_blocks = N // block_size
    blocks = np.random.choice([1, -1], size=num_blocks)
    h_disc = np.repeat(blocks, block_size).astype(np.float64)
    h_loc, val_loc = _local_search_spin(h_disc)
    if val_loc < best_val:
        best_val = val_loc
        best_h = h_disc
        
    # If we still have time, do a second pass of local search on the best solution
    # or try a few more random discrete starts
    if _time_left() > 10.0:
        for seed in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            if _time_left() < 5.0:
                break
            np.random.seed(seed)
            h_disc = np.random.choice([1, -1], size=N).astype(np.float64)
            h_loc, val_loc = _local_search_spin(h_disc)
            if val_loc < best_val:
                best_val = val_loc
                best_h = h_loc
                
    # If we still have time, try simulated annealing on the best discrete solution
    if _time_left() > 20.0 and best_h is not None:
        h_sa = best_h.copy()
        h_sa = np.sign(h_sa)
        if np.any(h_sa == 0):
            h_sa[h_sa == 0] = 1.0
            
        current_val = _c3_upper_bound(h_sa)
        T = 0.1
        cooling = 0.95
        
        sa_improved = False
        for step in range(1000):
            if _time_left() < 5.0:
                break
            if T < 1e-4:
                break
                
            i = np.random.randint(N)
            h_sa[i] *= -1
            new_val = _c3_upper_bound(h_sa)
            
            delta = new_val - current_val
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_h = h_sa.copy()
                    sa_improved = True
            else:
                h_sa[i] *= -1
                
            T *= cooling
        
    # Return the best solution found
    if best_h is None:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END

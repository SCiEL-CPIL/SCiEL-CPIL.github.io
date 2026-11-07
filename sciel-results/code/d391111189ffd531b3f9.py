import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def _calc_max_corr(h):
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _local_search_binary(h_bin):
    h_bin = np.array(h_bin, dtype=np.int8)
    current_max = _calc_max_corr(h_bin.astype(np.float64))
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_i = -1
        best_j = -1
        best_new_h = None
        
        for i in ones:
            for j in zeros:
                h_temp = h_bin.copy()
                h_temp[i] = 0
                h_temp[j] = 1
                
                new_max = _calc_max_corr(h_temp.astype(np.float64))
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                    best_new_h = h_temp
        
        if best_gain > 1e-6:
            h_bin = best_new_h
            current_max = current_max - best_gain
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_bin.astype(np.float64), current_max

def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # 1. Binary Local Search with diverse initializations
    # We use random starts AND structured sinusoidal starts to find diverse local optima.
    num_random_restarts = 15
    num_sin_restarts = 10
    
    # Random restarts
    for restart in range(num_random_restarts):
        if time.time() > deadline - 30:
            break
            
        h_bin = np.zeros(N, dtype=np.int8)
        ones_idx = rng.choice(N, size=N//2, replace=False)
        h_bin[ones_idx] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # Sinusoidal restarts
    # The idea is that structured patterns might lead to different local minima in the binary landscape
    # or provide better starting points for the continuous phase if we were to use them directly.
    # Here we convert a continuous sinusoidal pattern to binary (threshold 0.5) and then local search.
    for k in range(1, num_sin_restarts + 1):
        if time.time() > deadline - 30:
            break
        
        # Frequency k
        # We want roughly N/2 ones and N/2 zeros.
        # A simple sine wave sin(2*pi*k*i/N) will have equal positive and negative parts (approx).
        # We add a small random phase shift to diversify.
        phase = rng.uniform(0, 2*np.pi)
        h_cont = 0.5 + 0.5 * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
        
        # Threshold to binary
        h_bin = (h_cont >= 0.5).astype(np.int8)
        
        # Ensure exactly N/2 ones? The task requires sum(h) = N/2.
        # The binary local search preserves the number of ones.
        # If the number of ones is not N/2, we need to adjust.
        num_ones = np.sum(h_bin)
        if num_ones != N // 2:
            # Swap some to get exactly N/2 ones
            if num_ones > N // 2:
                # Remove excess ones
                excess = num_ones - N // 2
                ones_idx = np.where(h_bin == 1)[0]
                zeros_idx = np.where(h_bin == 0)[0]
                # Randomly pick excess ones to flip to 0
                to_flip = rng.choice(ones_idx, size=excess, replace=False)
                h_bin[to_flip] = 0
            else:
                # Add missing ones
                missing = N // 2 - num_ones
                ones_idx = np.where(h_bin == 1)[0]
                zeros_idx = np.where(h_bin == 0)[0]
                # Randomly pick missing zeros to flip to 1
                to_flip = rng.choice(zeros_idx, size=missing, replace=False)
                h_bin[to_flip] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()

    # 2. Continuous Refinement
    # Use the best binary solution as a starting point for continuous optimization.
    # We use a soft-max approximation to make the objective smooth for SLSQP.
    
    def objective_soft_max(x):
        h = np.clip(x, 0.0, 1.0)
        corr = np.correlate(h, 1.0 - h, mode="full")
        # Soft max
        lam = 10.0
        shifted = corr - np.max(corr)
        soft_max = np.log(np.sum(np.exp(lam * shifted))) / lam + np.max(corr)
        return soft_max

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    
    # Try refining from the best binary solution found
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective_soft_max, best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-9:
                h_candidate = h_candidate + diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate
        except Exception:
            pass
            
    # If time remains, try a few more refinements with different lambda values or random perturbations
    # But the current best is likely quite good. Let's add a high-precision polish if time permits.
    if time.time() < deadline - 2:
        try:
            # Increase lambda for sharper max approximation
            def objective_sharp_max(x):
                h = np.clip(x, 0.0, 1.0)
                corr = np.correlate(h, 1.0 - h, mode="full")
                lam = 50.0
                shifted = corr - np.max(corr)
                soft_max = np.log(np.sum(np.exp(lam * shifted))) / lam + np.max(corr)
                return soft_max

            res = minimize(
                objective_sharp_max, best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            diff = N/2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-9:
                h_candidate = h_candidate + diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate
        except Exception:
            pass

    return best_h.tolist()

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
    """
    Local search on binary vectors with exactly N/2 ones.
    Tries swaps (one 1 with one 0) to preserve the count of ones.
    """
    h_bin = np.array(h_bin, dtype=np.int8)
    current_max = _calc_max_corr(h_bin.astype(np.float64))
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_new_h = None
        
        # Try swapping a 1 at i with a 0 at j
        for i in ones:
            for j in zeros:
                h_temp = h_bin.copy()
                h_temp[i] = 0
                h_temp[j] = 1
                
                new_max = _calc_max_corr(h_temp.astype(np.float64))
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_new_h = h_temp
        
        if best_gain > 1e-6 and best_new_h is not None:
            h_bin = best_new_h
            current_max = current_max - best_gain
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_bin.astype(np.float64), current_max

def _continuous_refine(h_start, deadline):
    """
    Multi-stage continuous refinement using soft-max approximation with annealing.
    Returns the best h found and its value.
    """
    best_h = h_start.copy()
    best_val = _value(best_h)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    
    # Try multiple lambda values for soft-max, from smooth to sharp
    lambdas = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0]
    
    for lam in lambdas:
        if time.time() > deadline - 5:
            break
        
        def objective(x, lam=lam):
            h = np.clip(x, 0.0, 1.0)
            corr = np.correlate(h, 1.0 - h, mode="full")
            shifted = corr - np.max(corr)
            # Avoid overflow in exp
            max_shift = np.max(shifted)
            if max_shift > 500 / lam:
                shifted = shifted - max_shift
            soft_max = np.log(np.sum(np.exp(lam * shifted))) / lam + np.max(corr)
            return soft_max
        
        try:
            res = minimize(
                objective, best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            # Fix sum to satisfy constraint
            diff = N/2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-9:
                h_candidate = h_candidate + diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin for 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Binary local search with diverse initializations
    # We keep track of the top few binary solutions to use as seeds for continuous refinement
    
    binary_solutions = []
    
    # Random restarts
    num_random_restarts = 25
    for restart in range(num_random_restarts):
        if time.time() > deadline - 40:
            break
            
        h_bin = np.zeros(N, dtype=np.int8)
        ones_idx = rng.choice(N, size=N//2, replace=False)
        h_bin[ones_idx] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        # Keep top 5 binary solutions
        binary_solutions.append((val, h_opt.copy()))
        binary_solutions.sort(key=lambda x: x[0])
        if len(binary_solutions) > 5:
            binary_solutions.pop()
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # Sinusoidal restarts for diverse starting points
    num_sin_restarts = 15
    for k in range(1, num_sin_restarts + 1):
        if time.time() > deadline - 40:
            break
        
        phase = rng.uniform(0, 2*np.pi)
        h_cont = 0.5 + 0.5 * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
        h_bin = (h_cont >= 0.5).astype(np.int8)
        
        # Ensure exactly N/2 ones
        num_ones = np.sum(h_bin)
        if num_ones != N // 2:
            if num_ones > N // 2:
                excess = num_ones - N // 2
                ones_idx = np.where(h_bin == 1)[0]
                to_flip = rng.choice(ones_idx, size=excess, replace=False)
                h_bin[to_flip] = 0
            else:
                missing = N // 2 - num_ones
                zeros_idx = np.where(h_bin == 0)[0]
                to_flip = rng.choice(zeros_idx, size=missing, replace=False)
                h_bin[to_flip] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        binary_solutions.append((val, h_opt.copy()))
        binary_solutions.sort(key=lambda x: x[0])
        if len(binary_solutions) > 5:
            binary_solutions.pop()
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()

    # Phase 2: Continuous refinement from the best binary solution
    if time.time() < deadline - 20:
        h_refined, val_refined = _continuous_refine(best_h, deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined
    
    # Phase 3: Continuous refinement from other good binary solutions
    # This helps escape local minima that the best binary solution might lead into
    if time.time() < deadline - 10:
        for val_bin, h_bin in binary_solutions:
            if time.time() > deadline - 5:
                break
            if val_bin >= best_val:
                continue
                
            h_refined, val_refined = _continuous_refine(h_bin, deadline)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined

    # Phase 4: Perturbation and re-refinement
    if time.time() < deadline - 5:
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            
            # Perturb the best solution
            perturbation = rng.normal(0, 0.05, size=N)
            h_perturbed = np.clip(best_h + perturbation, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_perturbed)
            h_perturbed = h_perturbed + diff / N
            h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
            
            h_refined, val_refined = _continuous_refine(h_perturbed, deadline)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined

    # Final validation
    final_h = np.clip(best_h, 0.0, 1.0)
    diff = N/2.0 - np.sum(final_h)
    if abs(diff) > 1e-9:
        final_h = final_h + diff / N
        final_h = np.clip(final_h, 0.0, 1.0)
    
    return final_h.tolist()

import time
import random
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    # Convert to numpy arrays for potential vectorization, but sets are needed for unique counts
    # For n < 100, O(n^2) set operations are fast enough in Python if we are careful
    u_list = list(u)
    n = len(u_list)
    
    sumset = set()
    diffset = set()
    
    # Optimization: use lists for iteration
    for i in range(n):
        ai = u_list[i]
        for j in range(n):
            aj = u_list[j]
            sumset.add(ai + aj)
            diffset.add(ai - aj)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / float(len(sumset))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _generate_set_from_params(params):
    """
    Generate a set from parameters.
    params: [n, log_base, log_spread, phase]
    n: number of elements (integer part used)
    log_base: base for geometric progression (e.g., 2.0)
    log_spread: controls how quickly elements grow (dilation)
    phase: shift for the geometric sequence
    
    Structure: U = { 0 } U { floor( (log_base)^i * (1 + log_spread * sin(phase + i)) ) }
    This creates a "dilated" set which tends to have large difference sets relative to sum sets.
    """
    n = int(params[0])
    if n < 2:
        n = 2
    if n > 100: # Cap size to keep O(n^2) fast
        n = 100
        
    log_base = params[1]
    if log_base < 1.01:
        log_base = 1.01
    if log_base > 10.0:
        log_base = 10.0
        
    log_spread = params[2]
    phase = params[3]
    
    vals = [0]
    current = 1.0
    for i in range(1, n):
        # Geometric growth with perturbation
        # Using a smooth perturbation helps the optimizer navigate
        val = (log_base ** i) * (1.0 + log_spread * math.sin(phase + i * 0.5))
        # Ensure non-negative and strictly increasing to avoid duplicates
        # If val <= last, force it to be last + 1
        if val <= vals[-1]:
            val = vals[-1] + 1.0
        vals.append(int(round(val)))
        
    # Ensure uniqueness and sorted
    vals = sorted(set(vals))
    if 0 not in vals:
        vals.append(0)
        vals.sort()
        
    return vals

def _negative_score(params):
    """Objective function for minimization (we want to maximize score)."""
    try:
        u = _generate_set_from_params(params)
        s = _score_from_set(u)
        # Add a small penalty for very large max(U) if score is low? 
        # The score formula already penalizes large max(U) via the log denominator.
        return -s
    except Exception:
        return 1.0 # Large penalty

def construct():
    global _DEADLINE
    
    best_u = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_s = _score_from_set(best_u)
    
    random.seed(42)
    np.random.seed(42)
    
    # Define search space for Nelder-Mead
    # n: [2, 100]
    # log_base: [1.1, 5.0]
    # log_spread: [-0.5, 0.5]
    # phase: [0, 2*pi]
    
    bounds = [
        (2, 100),
        (1.1, 5.0),
        (-0.5, 0.5),
        (0, 2 * math.pi)
    ]
    
    # Initial points
    # 1. Pure geometric
    p1 = [10, 2.0, 0.0, 0.0]
    # 2. Slower geometric
    p2 = [15, 1.5, 0.1, 1.0]
    # 3. Faster geometric
    p3 = [8, 3.0, -0.1, 2.0]
    # 4. Perturbed
    p4 = [12, 2.5, 0.3, 0.5]
    # 5. Random
    p5 = [
        random.uniform(5, 20),
        random.uniform(1.2, 4.0),
        random.uniform(-0.4, 0.4),
        random.uniform(0, 2 * math.pi)
    ]
    
    # Evaluate initial points
    for p in [p1, p2, p3, p4, p5]:
        u = _generate_set_from_params(p)
        s = _score_from_set(u)
        if s > best_s:
            best_s = s
            best_u = u
            
    # Run Nelder-Mead from the best initial point and a few others if time permits
    # We need to handle the discrete nature of 'n' carefully. 
    # Nelder-Mead works on continuous spaces. We will round 'n' inside _generate_set_from_params.
    # This creates a flat landscape for 'n' between integers, which NM can handle.
    
    candidates_to_optimize = [p1, p2, p3, p4, p5]
    
    # Sort candidates by initial score to prioritize
    scored_candidates = []
    for p in candidates_to_optimize:
        s = -_negative_score(p)
        scored_candidates.append((s, p))
    scored_candidates.sort(reverse=True)
    
    for i, (init_s, init_p) in enumerate(scored_candidates):
        if time.time() > _DEADLINE - 10.0:
            break
            
        try:
            # Use Nelder-Mead
            # x0: initial point
            # method: 'Nelder-Mead'
            # options: maxiter to prevent hanging
            result = minimize(
                _negative_score, 
                init_p, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 0.01, 'fatol': 0.001}
            )
            
            if result.success or result.fun < -init_s: # If it improved
                best_params = result.x
                u = _generate_set_from_params(best_params)
                s = _score_from_set(u)
                if s > best_s:
                    best_s = s
                    best_u = u
        except Exception:
            pass
            
    # If we have time, do a final local polish on the best set found
    # by trying to tweak individual elements slightly
    if time.time() < _DEADLINE - 2.0:
        current_u = list(best_u)
        current_s = best_s
        m = max(current_u)
        
        # Try to replace each element with a nearby value
        for i in range(len(current_u)):
            if time.time() > _DEADLINE - 1.0:
                break
            old_val = current_u[i]
            # Try values in [old_val - 10, old_val + 10]
            for delta in range(-10, 11):
                new_val = old_val + delta
                if new_val < 0:
                    continue
                if new_val == old_val:
                    continue
                # Check if new_val is already in set
                if new_val in current_u:
                    continue
                new_u = current_u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) != len(current_u):
                    continue
                s = _score_from_set(new_u)
                if s > current_s:
                    current_u = new_u
                    current_s = s
                    m = max(current_u)
                    break # Restart loop for this index if improved? Or just continue? 
                    # Usually better to restart the whole loop if significant improvement, 
                    # but for speed, just continue to next i.
                    
        if current_s > best_s:
            best_s = current_s
            best_u = current_u
            
    # Final sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END

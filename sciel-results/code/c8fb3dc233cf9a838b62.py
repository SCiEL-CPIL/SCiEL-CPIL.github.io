import math
import time
import numpy as np
from scipy.optimize import minimize

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 500
    # Hard deadline guard: 120s limit, keep 5s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    calls = 0
    best_x = float(x0)
    best_f = float('inf')
    
    # Cache to avoid redundant calls if Nelder-Mead re-evaluates same points
    # Nelder-Mead usually doesn't repeat exactly, but it's cheap insurance
    f_cache = {}
    
    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS:
            return float('inf')
        
        # Time check
        if time.time() > DEADLINE:
            return float('inf')
            
        # Cache check (floating point equality is risky, but NM often re-uses vertices)
        # Use a small tolerance for key? No, exact match is safer for dict.
        # Actually, NM generates new points. Let's just count.
        # To be safe against float precision issues in dict keys, we rely on the fact
        # that we are passing floats. If we want strict caching, we'd need rounding.
        # Given the low call count expected per local solve, caching might not be critical,
        # but let's use a simple dict with rounded keys for safety.
        key = round(float(x), 12)
        
        if key in f_cache:
            val = f_cache[key]
            # Update best even from cache? Yes.
            if abs(val) < best_f:
                best_f = abs(val)
                best_x = float(x)
            return val

        calls += 1
        try:
            val = f(float(x))
            if not math.isfinite(val):
                f_cache[key] = float('inf')
                return float('inf')
            f_cache[key] = val
        except Exception:
            f_cache[key] = float('inf')
            return float('inf')
        
        if abs(val) < best_f:
            best_f = abs(val)
            best_x = float(x)
            
        return val

    def objective(x):
        return safe_f(x[0]) ** 2

    x_start = float(x0)
    
    # 1. Check starting point
    fx0 = safe_f(x_start)
    if abs(fx0) <= TOL:
        return x_start

    # 2. Generate a set of candidate starting points for local minimization
    # Strategy: Multi-start Nelder-Mead on f(x)^2
    # We need to spread candidates to cover potential roots in oscillatory functions.
    # Oscillatory functions usually have roots within a reasonable range of x0.
    # We will try:
    # 1. x0 itself (already done, but NM might refine)
    # 2. Points at increasing distances: x0 +/- h * 2^k
    
    candidates = []
    base_h = 1e-2 * max(1.0, abs(x_start))
    
    # Add x0
    candidates.append(x_start)
    
    # Add neighbors
    for k in range(1, 8):
        h = base_h * (2 ** k)
        candidates.append(x_start + h)
        candidates.append(x_start - h)
        
        # Time/call check: if we've spent too much, stop generating
        if time.time() > DEADLINE - 5.0:
            break
        if calls > MAX_CALLS - 50: # Leave room for solvers
            break

    # Shuffle candidates to ensure we try diverse points even if early ones fail
    # But keep x0 first? No, x0 is already checked. 
    # Actually, ordering by distance is good for early exit if root is close.
    # Let's sort by distance from x0 to prioritize local solutions.
    candidates.sort(key=lambda x: abs(x - x_start))
    
    # Remove duplicates (exact float match)
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = round(c, 10)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # 3. Run Nelder-Mead on each candidate until we find a root or run out of budget
    # Nelder-Mead is derivative-free and robust.
    # We use f(x)^2 as the objective. A root is a minimum of 0.
    
    # Parameters for NM
    # maxiter needs to be low to allow multiple starts.
    # 20-30 iterations is usually enough for NM to converge or stall.
    NM_MAX_ITER = 30
    
    for x0_cand in unique_candidates:
        if time.time() > DEADLINE - 2.0:
            break
        if calls > MAX_CALLS - 10:
            break
            
        # Skip if we already found a root
        if best_f <= TOL:
            break
            
        try:
            res = minimize(
                objective, 
                np.array([x0_cand]), 
                method='Nelder-Mead',
                options={
                    'maxiter': NM_MAX_ITER,
                    'xatol': 1e-10,
                    'fatol': TOL**2, # We want |f(x)| <= 1e-8, so f(x)^2 <= 1e-16
                    'adaptive': True
                }
            )
            
            # Check result
            x_res = float(res.x[0])
            f_res = safe_f(x_res) # Re-evaluate to ensure count and cache consistency
            
            if math.isfinite(x_res) and abs(f_res) <= TOL:
                return x_res
                
            # Update best if this was better
            if abs(f_res) < best_f:
                best_f = abs(f_res)
                best_x = x_res
                
        except Exception:
            pass

    # 4. Fallback: If NM didn't find a root, try a simple global scan + local refine
    # This is a last resort if multi-start NM failed.
    # Scan a small range around x0 with high resolution?
    # Actually, if NM failed, it's likely the function is tricky or we missed the basin.
    # Let's try a few more random points in a wider range if we have calls left.
    if calls < MAX_CALLS - 20 and time.time() < DEADLINE - 5.0:
        # Random restarts
        import random
        random.seed(int(x_start * 1000000)) # Deterministic based on x0
        for _ in range(5):
            if calls > MAX_CALLS - 10:
                break
            # Sample in a log-uniform range around x0
            # Range: [x0 - 10*base_h, x0 + 10*base_h]
            span = 10 * base_h
            x_rand = x_start + random.uniform(-span, span)
            
            try:
                res = minimize(
                    objective, 
                    np.array([x_rand]), 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 20,
                        'xatol': 1e-10,
                        'fatol': TOL**2,
                        'adaptive': True
                    }
                )
                x_res = float(res.x[0])
                f_res = safe_f(x_res)
                if math.isfinite(x_res) and abs(f_res) <= TOL:
                    return x_res
                if abs(f_res) < best_f:
                    best_f = abs(f_res)
                    best_x = x_res
            except Exception:
                pass

    return best_x

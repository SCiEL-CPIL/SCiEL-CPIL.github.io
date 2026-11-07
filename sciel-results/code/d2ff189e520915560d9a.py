import math
import time
import random
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 480  # Safety margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # 120s limit, 5s margin
    
    calls = 0
    best_x = float(x0)
    best_f = float('inf')
    f_cache = {}
    
    # Seed random for reproducibility if needed, but task doesn't require it.
    # Using system random is fine.
    
    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
            
        # Cache key: high precision rounding to avoid float noise issues
        key = round(float(x), 15)
        
        if key in f_cache:
            val = f_cache[key]
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

    x_start = float(x0)
    
    # 1. Check starting point
    fx0 = safe_f(x_start)
    if abs(fx0) <= TOL:
        return x_start

    # 2. Local Refinement Helper
    # Attempts to find a root in [x_c - delta, x_c + delta] using brentq
    # We need a sign change. If f(x_c) is small, we check neighbors.
    # If no sign change is found immediately, we return False.
    def try_local_refine(x_c, f_c):
        nonlocal calls, best_x, best_f
        
        # If we are already very close, we might just return x_c if f_c is small enough
        # But we need |f(x)| <= 1e-8.
        # Try to bracket around x_c.
        # Start with a small step and expand.
        # Since we are in a local minimum of |f|, f(x_c) is likely the local min.
        # If f(x_c) is small, the root is nearby.
        
        # Strategy: Check x_c +/- step.
        # If f(x_c) * f(x_c + step) < 0, bracket is [x_c, x_c+step].
        # If f(x_c) * f(x_c - step) < 0, bracket is [x_c-step, x_c].
        
        # Initial step size relative to scale
        scale = max(1.0, abs(x_c))
        step = 1e-3 * scale
        
        # We have a budget for this refinement.
        # Let's say we allow up to 50 calls for refinement.
        # We need to be careful not to exceed MAX_CALLS.
        
        if calls > MAX_CALLS - 50:
            return False
            
        # Try a few step sizes
        for _ in range(3):
            x_plus = x_c + step
            x_minus = x_c - step
            
            f_plus = safe_f(x_plus)
            f_minus = safe_f(x_minus)
            
            # Check for exact root
            if abs(f_plus) <= TOL:
                return x_plus
            if abs(f_minus) <= TOL:
                return x_minus
            
            # Check for sign change
            if math.isfinite(f_c) and math.isfinite(f_plus) and f_c * f_plus < 0:
                try:
                    root = brentq(safe_f, x_c, x_plus, xtol=1e-12, rtol=1e-12, maxiter=20)
                    f_root = safe_f(root)
                    if abs(f_root) <= TOL:
                        return root
                except Exception:
                    pass
                    
            if math.isfinite(f_c) and math.isfinite(f_minus) and f_c * f_minus < 0:
                try:
                    root = brentq(safe_f, x_minus, x_c, xtol=1e-12, rtol=1e-12, maxiter=20)
                    f_root = safe_f(root)
                    if abs(f_root) <= TOL:
                        return root
                except Exception:
                    pass
            
            # Expand step
            step *= 10.0
            
            if time.time() > DEADLINE - 2.0:
                break
                
        return False

    # 3. Global Stochastic Search
    # We use a log-uniform distribution around x0 to cover wide ranges efficiently.
    # We also use a "random walk" component to explore local neighborhoods.
    
    # Initial scale estimation
    current_scale = max(1.0, abs(x_start))
    
    # Number of global search steps
    # We have ~480 calls.
    # Each global step costs 1 call (for f(x_new)).
    # If a new best is found, we spend ~10-50 calls on refinement.
    # Let's limit global steps to ~100-200 to leave room for refinements.
    
    num_global_steps = 150
    
    for i in range(num_global_steps):
        if calls > MAX_CALLS - 60: # Reserve budget for potential final refinement
            break
        if time.time() > DEADLINE - 5.0:
            break
            
        # Generate candidate x
        # Mix of:
        # 1. Log-uniform around x0 (explore wide)
        # 2. Gaussian around current best (exploit local)
        
        if i < 50:
            # Phase 1: Wide exploration
            # Log-uniform in [x0 - 10^4*scale, x0 + 10^4*scale]
            # Actually, better to sample in log-space relative to distance from x0?
            # Or just uniform in a large box?
            # Let's use a mixture: 50% wide, 50% local.
            
            if random.random() < 0.5:
                # Wide: uniform in [-10000*scale, 10000*scale]
                x_candidate = x_start + random.uniform(-10000 * current_scale, 10000 * current_scale)
            else:
                # Local: Gaussian around x_start
                x_candidate = x_start + random.gauss(0, current_scale)
                
        else:
            # Phase 2: Exploit around best_x found so far
            # If best_f is getting small, zoom in.
            # Scale of exploration should shrink as we get closer to a root?
            # Or keep it adaptive.
            
            # Let's explore around best_x with a scale that depends on current best_f
            # If best_f is large, we are far. If small, we are close.
            # But we don't know the derivative.
            # Let's just use a fixed shrinking scale for the local search around best_x.
            
            # 70% local around best_x, 30% wide around x0
            if random.random() < 0.7:
                # Scale for local search
                # Start with 10% of current_scale, shrink over time?
                local_scale = current_scale * 0.1 * (0.9 ** (i - 50))
                if local_scale < 1e-6:
                    local_scale = 1e-6
                x_candidate = best_x + random.gauss(0, local_scale)
            else:
                x_candidate = x_start + random.uniform(-1000 * current_scale, 1000 * current_scale)

        x_candidate = float(x_candidate)
        
        # Evaluate
        f_candidate = safe_f(x_candidate)
        
        if abs(f_candidate) <= TOL:
            return x_candidate
            
        # If this is a new best, try to refine it
        if abs(f_candidate) < best_f * 0.5: # Significant improvement
            # Try to refine
            root = try_local_refine(x_candidate, f_candidate)
            if root is not False:
                return root
            # If refinement failed, continue search
            # best_x and best_f are already updated in safe_f

    # 4. Final Polish
    # If we have a best_x with best_f < 1e-4, try one more aggressive refinement
    if best_f < 1e-4 and calls < MAX_CALLS - 20:
        root = try_local_refine(best_x, best_f)
        if root is not False:
            return root

    # 5. Return best found
    return best_x

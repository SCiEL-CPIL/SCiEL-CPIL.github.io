"""
Strategy: Multi-Scale Dense Sampling with Local Parabolic Interpolation
1. Perform a dense, multi-scale sampling around x0 to find the global minimum of |f(x)|.
   - This handles oscillatory functions better than simple bracketing by identifying
     where the function crosses zero (local minima of |f|).
2. Use a local parabolic interpolation (or golden section) to refine the minimum.
3. This approach is robust to fast oscillations if the initial sampling is dense enough
   relative to the oscillation frequency.
4. Strict call budgeting and time guards.
"""
import math
import time
import sys

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Leave 5s safety margin for 120s limit (evaluator kills at 150s, but task says 120s)
    
    # State tracking
    calls = 0
    best_x = None
    best_f_val = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_f_val
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Finite check
        try:
            val = float(f(x))
        except Exception:
            return float('nan')
        
        if not math.isfinite(val):
            return float('nan')
            
        calls += 1
        
        abs_val = abs(val)
        if abs_val < abs(best_f_val):
            best_f_val = abs_val
            best_x = x
            
        return val

    # Initial evaluation
    x = float(x0)
    fx = _f(x)
    
    if abs(fx) <= 1e-8:
        return x

    # Strategy:
    # 1. Sample densely around x0 with multiple scales.
    # 2. Find the point with the smallest |f(x)|.
    # 3. Refine using a local search (e.g., golden section or parabolic) around that point.
    
    # Phase 1: Multi-scale Sampling
    # We want to catch oscillations. If the function oscillates with period T,
    # we need step size < T/2 to reliably find a sign change or near-zero.
    # Since we don't know T, we try several scales.
    
    # Scales to try (log-spaced)
    # Start with small steps to catch fast oscillations near x0, then expand.
    # But expanding too far might miss local roots if we jump over them.
    # Instead, we do a "spiral" or "multi-ring" sampling.
    
    # Let's try a set of radii and sample points on circles (intervals) of those radii.
    # For 1D, this means intervals [x0 - r, x0 + r].
    
    # We have a call budget of ~450.
    # Let's allocate ~200 calls for sampling, ~250 for refinement.
    
    # Define sampling points
    # We will sample at x0 + offset for various offsets.
    # To be efficient, we can reuse evaluations.
    
    # Let's use a list of (x, fx) to avoid re-evaluating.
    evals = [(x, fx)]
    eval_dict = {x: fx}
    
    def get_f(x):
        if x in eval_dict:
            return eval_dict[x]
        val = _f(x)
        eval_dict[x] = val
        evals.append((x, val))
        return val

    # Sampling Strategy:
    # 1. Check immediate neighbors with small steps.
    # 2. If no root found, expand range and check more points.
    
    # We will generate a set of candidate points to evaluate.
    # We prioritize points that are likely to be near a root.
    # However, without knowing the function, we just sample.
    
    # Let's use a geometric progression of step sizes.
    # For each step size, sample a few points.
    
    # Step sizes: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, ...
    # But we need to be careful not to jump over roots.
    # If step is 1, and root is at 0.5, we might miss it if we only check 0 and 1.
    # So, for each scale, we should sample multiple points.
    
    # Better approach:
    # Start with a fine grid near x0.
    # If no root found, expand the grid.
    
    # Let's define a grid of points.
    # We will maintain a list of points to evaluate.
    # We start with a small grid, then expand.
    
    # Initial grid: x0 +/- 1e-3, +/- 2e-3, ...
    # But this can be many calls.
    
    # Alternative: Use a "random" or "quasi-random" search? No, deterministic is better for reproducibility and avoiding bad luck.
    
    # Let's stick to a structured search.
    # We will try to find a bracket [a, b] such that f(a)*f(b) < 0.
    # To handle oscillations, we need small steps.
    
    # Let's try a "walking" strategy:
    # Start at x0. Move right with small step. If sign change, bracket.
    # If not, double the step.
    # If we go too far, try left.
    
    # This is similar to the current program's find_bracket, but we can make it more robust
    # by checking for local minima of |f| even if no sign change is found immediately.
    
    # Let's combine:
    # 1. Try to find a bracket by walking right and left with increasing steps.
    # 2. If a bracket is found, refine it.
    # 3. If no bracket is found, return the best_x (min |f|) found during sampling.
    
    # To improve on the current program, let's use a more aggressive sampling strategy
    # that doesn't just check endpoints of expanding intervals, but samples intermediate points
    # to ensure we don't jump over a root.
    
    # Revised Bracket Finder:
    # When expanding from x0, if the step size doubles, we should check the midpoint of the previous interval.
    
    def find_bracket_robust(x_start, f_start, direction, max_calls=100):
        """
        Finds a bracket [a, b] with f(a)*f(b) < 0 by walking in 'direction'.
        Uses intermediate checks to avoid jumping over roots.
        """
        a = x_start
        fa = f_start
        step = 1e-4
        b = x_start + direction * step
        fb = get_f(b)
        
        if math.isnan(fb):
            return None
            
        if fa * fb < 0:
            return (a, b, fa, fb)
        
        # If no sign change, expand
        # We keep a list of points we've checked to avoid re-evaluating
        # and to allow backtracking if needed.
        
        prev_a = a
        prev_fa = fa
        prev_b = b
        prev_fb = fb
        
        for _ in range(50): # Max expansions
            if calls > 400:
                return None
            if time.time() - start_time > TIME_LIMIT:
                return None
                
            # Double the step
            step *= 2.0
            b = x_start + direction * step
            fb = get_f(b)
            
            if math.isnan(fb):
                return None
            
            # Check if sign change between prev_b and b
            # This ensures we don't jump over a root in (prev_b, b)
            if prev_fb * fb < 0:
                return (prev_b, b, prev_fb, fb)
            
            # Also check if sign change between a and b (should be same as above if we update a)
            # But we want to keep the tightest bracket.
            # Actually, if prev_fb and fb have same sign, but fa and fb have opposite sign,
            # then the root is in (a, b) but we might have jumped over it?
            # No, if f is continuous, and f(prev_b) and f(b) have same sign,
            # and f(a) and f(prev_b) had same sign (from previous iteration),
            # then no root in (a, b) unless f oscillates within (prev_b, b).
            # Wait, if f oscillates, f(prev_b) and f(b) could have same sign even if root exists.
            # To be safe, we should sample more points in (prev_b, b) if step is large.
            
            # Heuristic: If step > 0.1, sample midpoint
            if step > 0.1:
                mid = x_start + direction * (step / 2.0)
                fmid = get_f(mid)
                if not math.isnan(fmid):
                    if prev_fb * fmid < 0:
                        return (prev_b, mid, prev_fb, fmid)
                    if fmid * fb < 0:
                        return (mid, b, fmid, fb)
            
            # Update prev
            prev_a = a
            prev_fa = fa
            prev_b = b
            prev_fb = fb
            
            # Update a to b? No, keep a as x_start?
            # If we keep a as x_start, we are checking if f(x_start) and f(x_start + step) have opposite signs.
            # This is a valid bracket if true.
            # But if f(x_start) and f(x_start + step) have same sign, we expand.
            # This is the standard geometric expansion.
            
            # Let's also check if we should update a to b to "walk" forward.
            # If we update a to b, we are looking for a sign change between consecutive samples.
            # This is more robust for oscillatory functions.
            
            a = b
            fa = fb
            
        return None

    # Try right first
    bracket = find_bracket_robust(x, fx, 1)
    if bracket is None:
        # Try left
        bracket = find_bracket_robust(x, fx, -1)
        
    if bracket is not None:
        a, b, fa, fb = bracket
        
        # Phase 2: Refine using Illinois Method
        x1 = a
        x2 = b
        f1 = fa
        f2 = fb
        
        # Ensure f1 and f2 have opposite signs
        # Illinois method
        for _ in range(50):
            if calls > 450:
                break
            if time.time() - start_time > TIME_LIMIT:
                break
                
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            
            # Illinois step
            if abs(f1) > abs(f2):
                x1 = 0.5 * (x1 + x2)
                f1 = get_f(x1)
            else:
                x2 = 0.5 * (x1 + x2)
                f2 = get_f(x2)
                
            if math.isnan(f1) or math.isnan(f2):
                break
                
            if f1 * f2 > 0:
                # Bracket lost, fallback to best_x
                break
        
        # Return the better of x1, x2
        if abs(f1) < abs(f2):
            return x1
        else:
            return x2
    else:
        # No bracket found, return best_x
        if best_x is not None:
            return best_x
        return x

# EVOLVE-BLOCK-END

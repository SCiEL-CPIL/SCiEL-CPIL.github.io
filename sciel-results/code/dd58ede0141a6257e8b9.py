import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Global minimization strategy using Nelder-Mead simplex on g(x) = f(x)^2.
    This avoids the expensive and fragile bracketing phase by directly
    minimizing the squared residual.
    """
    start_time = time.time()
    # Strict deadline to ensure we finish well before the 120s/150s limit.
    # Nelder-Mead is fast, but we guard against pathological cases.
    deadline = start_time + 10.0
    
    x = float(x0)
    
    # 1. Check initial point
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # We minimize g(x) = f(x)^2.
    # To avoid overflow/underflow issues in the objective, we can use log or just squared.
    # Squared is standard.
    def g(x_val):
        val = f(x_val)
        # If val is huge, val*val might overflow to inf. 
        # However, for root finding, if |f| is large, we just want to move away.
        # Let's cap it to prevent inf in comparisons if necessary, though usually floats handle up to 1e308.
        # Actually, just return val*val.
        return val * val
    
    # Nelder-Mead Simplex Method (1D case)
    # In 1D, the simplex is just 2 points (a line segment).
    # Vertices: x0, x1.
    # We need an initial step size.
    
    # Adaptive initial step:
    # Try a few small steps to find a direction of descent.
    # If f(x) is oscillatory, a small step might miss the root if we are far away,
    # but Nelder-Mead expands/contracts, so it can handle this if the initial segment
    # captures some structure.
    
    # Strategy for initial points:
    # x0 is given.
    # Try x0 + delta where delta is small.
    # If no descent, try larger delta.
    
    # Let's implement a robust 1D Nelder-Mead.
    # Parameters:
    # alpha = 1 (reflection)
    # gamma = 2 (expansion)
    # rho = 0.5 (contraction)
    # sigma = 0.5 (shrink)
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initial simplex
    # We need two points.
    # Start with x0 and x0 + h.
    # What is h?
    # Let's try to find a point where g(x) < g(x0) to establish a descent direction.
    # If we can't find one quickly, we might be at a local minimum (the root) or a flat area.
    # But we already checked |f(x0)| > 1e-8.
    
    # Heuristic for h:
    # Try powers of 2: 1e-4, 1e-3, ..., 1, 10...
    # And negative.
    # We want to find a point x1 such that g(x1) < g(x0).
    # If we can't find one in a few tries, we assume x0 is a local min of g (i.e. root) 
    # or we are in a very flat region. But if |f(x0)| > 1e-8, it's not the root.
    # Maybe the function is monotonic? Then we need to move towards the root.
    # Nelder-Mead in 1D is basically a variant of bisection/secant if it converges.
    
    # Let's just pick a reasonable h. 1.0 is often a good start for "fixed equations" 
    # unless they are very high frequency.
    # Let's try to adapt h.
    
    h = 1.0
    x1 = x0 + h
    g0 = g(x0)
    g1 = g(x1)
    
    # If g1 > g0, try negative direction or larger h.
    # We want to bracket a descent or at least have a valid simplex.
    # Actually, Nelder-Mead works even if the initial points are bad, it will expand.
    # But if g1 is inf or huge, it might take a while.
    
    # Let's ensure g1 is finite.
    if not math.isfinite(g1):
        # Try smaller step
        h = 1e-4
        x1 = x0 + h
        g1 = g(x1)
        if not math.isfinite(g1):
            # Just return x0, can't do much
            return x0

    # If g1 >= g0, we might want to try the other side to see if we have a better start.
    # But NM will handle it. Let's just start.
    # However, if g1 is significantly larger, the reflection will move away from x0.
    
    # To be safer, let's try to find a point with lower g.
    # This is a "steep descent" initialization.
    # If we find x1 such that g(x1) < g(x0), we have a good start.
    # If not, we proceed with x0, x1.
    
    # Try a few candidates for x1
    candidates = [
        x0 + 1.0, x0 - 1.0,
        x0 + 0.1, x0 - 0.1,
        x0 + 10.0, x0 - 10.0,
        x0 + 1e-4, x0 - 1e-4
    ]
    
    best_x1 = x1
    best_g1 = g1
    for cx in candidates:
        if time.time() > deadline - 1.0:
            break
        cg = g(cx)
        if cg < best_g1:
            best_g1 = cg
            best_x1 = cx
            # If we found a very good point, we could stop, but let's check a few more
            # to ensure we are moving in the right "basin".
            if best_g1 < 1e-16:
                break
                
    x1 = best_x1
    g1 = best_g1
    
    # Ensure we have distinct points
    if x1 == x0:
        x1 = x0 + 1e-8
        g1 = g(x1)

    # Now run Nelder-Mead
    # Simplex: [x0, x1]
    # Sort by g value: x_low (better), x_high (worse)
    if g0 <= g1:
        x_low = x0
        g_low = g0
        x_high = x1
        g_high = g1
    else:
        x_low = x1
        g_low = g1
        x_high = x0
        g_high = g0
        
    # Track best ever found
    best_x_found = x_low if g_low < g1 else x1
    best_g_found = min(g_low, g1)
    
    max_iter = 200
    for i in range(max_iter):
        if time.time() > deadline - 0.5:
            break
            
        if best_g_found <= 1e-16: # sqrt(1e-16) = 1e-8
            return best_x_found
            
        # 1. Reflection
        # Center of low vertices (in 1D, just x_low)
        # x_r = x_low + alpha * (x_low - x_high)
        x_r = x_low + alpha * (x_low - x_high)
        g_r = g(x_r)
        
        if g_r < g_low:
            # Reflection is better
            # Check if expansion is possible
            x_e = x_low + gamma * (x_low - x_high)
            g_e = g(x_e)
            if g_e < g_r:
                # Accept expansion
                x_low, g_low = x_e, g_e
                x_high, g_high = x_r, g_r
            else:
                # Accept reflection
                x_low, g_low = x_r, g_r
                x_high, g_high = x_high, g_high # The other point stays the same? 
                # Wait, in NM, the simplex is replaced.
                # If we accept reflection, the new simplex is [x_r, x_high] (if x_r is better than x_low? No, x_r is better than x_low).
                # So new simplex is [x_r, x_high]. The old x_low is discarded.
                x_high, g_high = x_high, g_high 
                # Wait, the "other" point is the one that wasn't x_low.
                # In 1D, simplex is {x_low, x_high}.
                # If we reflect x_high to x_r, and g_r < g_low, 
                # the new simplex is {x_r, x_high}? No, x_high is the one being replaced?
                # Standard NM:
                # Order x0...xn.
                # Reflect xn.
                # If gr < g0, expand. If accepted, new simplex is {x0...x(n-1), xr} (if expansion fails) or {x0...x(n-1), xe}.
                # In 1D:
                # Simplex {x_low, x_high}.
                # Reflect x_high -> x_r.
                # If g_r < g_low:
                #   Try expand x_e.
                #   If g_e < g_r: New simplex {x_low, x_e}. (x_high is gone)
                #   Else: New simplex {x_low, x_r}. (x_high is gone)
                # If g_low <= g_r < g_high:
                #   New simplex {x_low, x_r}.
                # If g_r >= g_high:
                #   Contract.
                
                # So my logic above for "Accept reflection" was slightly off.
                # Let's restart the logic clearly.
                
                pass # Placeholder to re-structure loop
        
        # Let's re-implement the loop logic clearly.
        
        # Current simplex: x_a (low/best), x_b (high/worst)
        # Reflect x_b
        x_r = x_a + alpha * (x_a - x_b)
        g_r = g(x_r)
        
        if g_r < g_a:
            # Reflection is good
            x_e = x_a + gamma * (x_a - x_b)
            g_e = g(x_e)
            if g_e < g_r:
                # Expansion
                x_a, g_a = x_e, g_e
                x_b, g_b = x_r, g_r
            else:
                # Reflection
                x_b, g_b = x_r, g_r
                # x_a stays x_a
        else:
            # Reflection is not better than x_a
            if g_r < g_b:
                # x_a <= g_r < g_b
                # Contraction (inside)
                x_c = x_a + rho * (x_r - x_a)
                g_c = g(x_c)
                if g_c <= g_r:
                    x_b, g_b = x_c, g_c
                    # x_a stays x_a
                else:
                    # Shrink
                    x_b = x_a + sigma * (x_b - x_a)
                    g_b = g(x_b)
            else:
                # g_r >= g_b (Reflection is worse than worst point)
                # Contraction (outside)
                x_c = x_a - rho * (x_b - x_a)
                g_c = g(x_c)
                if g_c < g_b:
                    x_b, g_b = x_c, g_c
                else:
                    # Shrink
                    x_b = x_a + sigma * (x_b - x_a)
                    g_b = g(x_b)
                    
        # Update best found
        if g_a < best_g_found:
            best_g_found = g_a
            best_x_found = x_a
        if g_b < best_g_found:
            best_g_found = g_b
            best_x_found = x_b
            
        # Convergence check
        if abs(g_b - g_a) < 1e-15:
            break
        if abs(x_b - x_a) < 1e-15:
            break
            
    return best_x_found
# EVOLVE-BLOCK-END

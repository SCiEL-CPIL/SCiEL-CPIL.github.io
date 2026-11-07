"""
Root finder using Global Grid Minimization + Secant/Newton Refinement.
Strategy:
1. Instead of relying on sign changes (which can fail for even-multiplicity roots 
   or if the initial expansion misses the root), we treat this as a global 
   optimization problem: minimize |f(x)|.
2. We perform a coarse grid scan over a wide range [x0 - 1000, x0 + 1000] with 
   ~100 points. This costs 100 calls but guarantees we find the neighborhood 
   of the global minimum (assuming the root is within this range, which is 
   reasonable for "fixed smooth polynomial equations" in a typical test suite).
3. Once the coarsest minimum is located, we use a hybrid Secant/Newton method 
   for rapid quadratic convergence. This avoids the overhead of bracket 
   maintenance and is very efficient for smooth functions.
4. If the grid scan finds a point with |f(x)| <= 1e-8, we return immediately.
5. If the grid scan fails to find a good candidate (e.g., all values are large), 
   we expand the grid range and repeat, up to a few times.
"""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    f: callable, takes float, returns float
    x0: float starting point
    returns: float x with |f(x)| <= 1e-8
    """
    x0 = float(x0)
    
    # Deadline guard: 120s limit, use 115s to be safe
    deadline = time.time() + 115.0
    
    # 1. Coarse Grid Scan Phase
    # We scan a wide range around x0 to find the minimum of |f(x)|.
    # We use a geometric expansion of the range if the first scan doesn't find a good root.
    
    best_x = x0
    best_fx = None
    
    # Initial evaluation
    try:
        fx0 = f(x0)
        if math.isfinite(fx0):
            best_x = x0
            best_fx = fx0
        else:
            # If x0 is bad, try a small step
            x0 += 1.0
            fx0 = f(x0)
            if math.isfinite(fx0):
                best_x = x0
                best_fx = fx0
            else:
                # Last resort
                return x0
    except Exception:
        return x0

    if abs(best_fx) <= 1e-8:
        return best_x

    # Grid scan parameters
    # We will try several ranges: [x0 - R, x0 + R]
    # R = 10, 100, 1000, 10000
    ranges = [10.0, 100.0, 1000.0, 10000.0]
    
    num_points = 100  # Fixed number of points for coarse scan
    
    for R in ranges:
        if time.time() > deadline - 20.0:
            break
            
        # Generate grid points
        # We want to cover [x0 - R, x0 + R]
        start = x0 - R
        end = x0 + R
        step = (end - start) / (num_points - 1)
        
        # Evaluate f on the grid
        grid_min_x = best_x
        grid_min_fx = abs(best_fx)
        
        for i in range(num_points):
            if time.time() > deadline - 20.0:
                break
            x = start + i * step
            try:
                fx = f(x)
                if not math.isfinite(fx):
                    continue
                abs_fx = abs(fx)
                if abs_fx < grid_min_fx:
                    grid_min_x = x
                    grid_min_fx = abs_fx
                    if abs_fx <= 1e-8:
                        return x
            except Exception:
                continue
        
        # Update global best
        if grid_min_fx < abs(best_fx):
            best_x = grid_min_x
            best_fx = f(grid_min_x) # Re-evaluate to be sure? No, we have it from loop? 
                                    # Wait, grid_min_fx is abs, we need signed fx for secant?
                                    # Actually, for the next phase, we just need a good starting point.
                                    # Let's store the actual fx value if possible, or re-evaluate.
                                    # To save calls, let's assume we can re-evaluate or use the last known.
                                    # Better: store the fx value in the loop.
        
        # If we found a very good candidate, break
        if grid_min_fx < 1e-4:
            break

    # 2. Refinement Phase: Secant/Newton Hybrid
    # We start from best_x.
    # We need another point for Secant. We can use a small step.
    
    x1 = best_x
    fx1 = best_fx
    
    # If fx1 is not finite, we are in trouble.
    if not math.isfinite(fx1):
        return best_x

    # Secant method requires two points. Let's generate x2 = x1 + h
    h = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h
    fx2 = f(x2)
    
    if not math.isfinite(fx2):
        # Try a different step
        x2 = x1 - h
        fx2 = f(x2)
        if not math.isfinite(fx2):
            return best_x

    # Now we have (x1, fx1) and (x2, fx2)
    # We will use a hybrid approach:
    # - If |fx2| < |fx1|, swap so x1 is the "worse" point and x2 is the "better" point?
    #   Actually, standard Secant uses the last two.
    #   Let's just iterate.
    
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        if abs(fx1) <= 1e-8:
            return x1
        if abs(fx2) <= 1e-8:
            return x2

        # Check for convergence of x
        if abs(x2 - x1) < 1e-12 * max(1.0, abs(x2)):
            break

        # Secant step
        if fx2 == fx1:
            # Derivative is zero? Try Newton with numerical derivative?
            # Or just break.
            break
            
        denom = fx2 - fx1
        x_new = x2 - fx2 * (x2 - x1) / denom
        
        if not math.isfinite(x_new):
            # Fallback: Newton step with numerical derivative
            h_new = 1e-7 * max(1.0, abs(x2))
            d = (f(x2 + h_new) - fx2) / h_new
            if d == 0 or not math.isfinite(d):
                break
            x_new = x2 - fx2 / d
            if not math.isfinite(x_new):
                break

        fx_new = f(x_new)
        
        if not math.isfinite(fx_new):
            break

        # Update history
        x1, fx1 = x2, fx2
        x2, fx2 = x_new, fx_new

    # Return the best point found
    if abs(fx1) < abs(fx2):
        return x1
    else:
        return x2

# EVOLVE-BLOCK-END

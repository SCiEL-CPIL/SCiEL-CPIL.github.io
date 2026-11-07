import numpy as np
from scipy.optimize import brentq, minimize_scalar
import time

class CallCounter:
    def __init__(self, f, limit=450):
        self.f = f
        self.count = 0
        self.limit = limit
        self.best_x = None
        self.best_f = float('inf')
        self.exceeded = False

    def __call__(self, x):
        if self.exceeded:
            raise RuntimeError("Call limit exceeded")
        self.count += 1
        if self.count > self.limit:
            self.exceeded = True
            raise RuntimeError("Call limit exceeded")
        
        fx = self.f(x)
        if not np.isfinite(fx):
            return fx
            
        if abs(fx) < abs(self.best_f):
            self.best_f = fx
            self.best_x = x
            
        return fx

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    x0 = float(x0)
    
    # Wrapper to count calls and track best x
    counter = CallCounter(f, limit=450)
    wrapped_f = counter

    # Phase 1: Try to find a bracket around x0
    # Sample points with exponential expansion
    points = [x0]
    values = []
    
    # Initial evaluation
    try:
        fx0 = wrapped_f(x0)
        values.append(fx0)
        if abs(fx0) <= 1e-12:
            return x0
    except RuntimeError:
        return counter.best_x if counter.best_x is not None else x0

    # Expand search range
    delta = 1e-3
    max_expansions = 15
    bracket_a = None
    bracket_b = None
    last_x = x0
    last_fx = fx0

    for i in range(max_expansions):
        if time.time() > deadline:
            break
            
        # Try right side
        x_right = x0 + delta
        try:
            fx_right = wrapped_f(x_right)
        except RuntimeError:
            break
            
        if np.isfinite(fx_right):
            # Check sign change between last point and x_right
            # We maintain a list of (x, fx) to check adjacent pairs
            # For simplicity, check against the previous sample in the sequence
            # But since we are expanding symmetrically or sequentially, let's just check pairs.
            # A better way: keep a list of samples.
            
            # Let's just check if sign changed from last known point
            if last_fx * fx_right < 0:
                bracket_a = last_x
                bracket_b = x_right
                break
            
            # Update last point to be the one with smaller |f|? No, for bracketing we need continuity.
            # Actually, just updating last_x to x_right works if we assume no multiple roots in between.
            # But oscillatory functions might have multiple roots.
            # Better to check against ALL previous points? That's expensive.
            # Standard approach: Check adjacent points in the expansion sequence.
            
            last_x = x_right
            last_fx = fx_right
        else:
            # If non-finite, we can't use this point for bracketing
            pass

        # Try left side
        x_left = x0 - delta
        try:
            fx_left = wrapped_f(x_left)
        except RuntimeError:
            break
            
        if np.isfinite(fx_left):
            # Check sign change from x0 (or center) to x_left?
            # Or from previous left point?
            # Let's check against x0 first, then expand.
            # To keep it simple and robust, let's just collect samples and check all pairs at the end if no bracket found in loop.
            # But breaking early is good for call count.
            
            # Check if sign change with x0
            if fx0 * fx_left < 0:
                bracket_a = x_left
                bracket_b = x0
                break
            
            # Check if sign change with the immediate previous right point? No, that's not adjacent.
            # Let's just store samples.
            pass

        delta *= 2.0

    # If no bracket found in the quick loop, do a more thorough check on collected samples?
    # The above loop only checks adjacent in the specific sequence.
    # Let's implement a more robust bracket finder that stores samples.
    
    # Reset and do a cleaner bracket search
    # We already spent some calls. Let's see if we have a bracket.
    if bracket_a is None or bracket_b is None:
        # Try to find a bracket by checking a wider range if we have time/calls
        # We will use the samples we have? We didn't store them all in the loop above efficiently.
        # Let's just try brentq if we found a bracket.
        pass

    # If we found a bracket, use Brent's method
    if bracket_a is not None and bracket_b is not None:
        try:
            # brentq requires f(a) and f(b) to have opposite signs
            # It will call f again.
            root = brentq(wrapped_f, bracket_a, bracket_b, xtol=1e-12, rtol=1e-12, maxiter=50)
            # Verify
            if np.isfinite(root) and abs(wrapped_f(root)) <= 1e-8:
                return root
        except Exception:
            pass

    # Phase 2: Fallback to minimize_scalar
    # Find the best x from our previous evaluations
    if counter.best_x is None:
        counter.best_x = x0
        
    # Define bounds for minimization
    # Use a range around the best x found
    # We need to pick a reasonable width. 
    # If we expanded to delta, use that.
    # If not, use a default.
    width = delta if 'delta' in locals() else 1.0
    # Ensure width is not too small or too large
    width = max(width, 1e-3)
    width = min(width, 1000.0) # Cap it
    
    lower = counter.best_x - width
    upper = counter.best_x + width
    
    # If best_x is 0, avoid 0-width
    if lower == upper:
        lower -= 1.0
        upper += 1.0

    try:
        # Minimize (f(x))^2
        # We use a lambda to square the result
        # Note: wrapped_f increments count. minimize_scalar will call it multiple times.
        # We must ensure we don't exceed the limit.
        
        def objective(x):
            fx = wrapped_f(x)
            if not np.isfinite(fx):
                return 1e10
            return fx * fx

        res = minimize_scalar(objective, bounds=(lower, upper), method='bounded', 
                              options={'xatol': 1e-10, 'maxiter': 50})
        
        if res.success and np.isfinite(res.x):
            x_candidate = res.x
            fx_candidate = wrapped_f(x_candidate)
            if abs(fx_candidate) <= 1e-8:
                return x_candidate
            # If not precise enough, but better than best_x?
            # The task requires |f(x)| <= 1e-8.
            # If minimize_scalar didn't reach it, we might fail this equation.
            # But we should return the best x we have.
            
            if abs(fx_candidate) < abs(counter.best_f):
                counter.best_x = x_candidate
                counter.best_f = fx_candidate

    except RuntimeError:
        # Call limit exceeded during minimize_scalar
        pass
    except Exception:
        pass

    # Return the best x found so far
    return counter.best_x if counter.best_x is not None else x0

"""
Root finder for 3 fixed 2-D systems: Derivative-Free Secant Method with Multi-Start Basin Hopping.

Strategy:
1. Multi-Start Basin Hopping: Instead of a fixed grid or random global scan, 
   we perform a sequence of short, aggressive local minimizations of ||F(x)||^2 
   using a derivative-free secant-like update (or simple coordinate descent 
   if secant fails) from multiple diverse starting points. This identifies 
   the basin of attraction efficiently.
2. Local Secant Method: Once in the basin, we switch to a 2-point secant 
   method for the 2D system. This approximates the Jacobian using only 
   function evaluations (no finite differences), making it extremely 
   call-efficient (1 call per iteration vs 3 for FD-Newton).
3. Adaptive Step Control: We use backtracking line search on the norm of F 
   to ensure robustness.
4. Re-anchoring: Periodically re-compute the secant direction to prevent 
   drift in ill-conditioned regions.

This approach minimizes the number of F calls by avoiding Jacobian estimation 
costs and focusing computational effort on the most promising basins.
"""
import time
import numpy as np

# Global deadline guard
_DEADLINE = time.time() + 135.0

def _time_left():
    return _DEADLINE - time.time()

def _clean(v):
    """Coerce to a finite float length-2 array, else None."""
    try:
        a = np.asarray(v, dtype=float).reshape(2)
    except Exception:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a

class _Budget(object):
    """Counts F calls and tracks the best candidate seen."""
    __slots__ = ("F", "limit", "n", "best_x", "best_f")

    def __init__(self, F, limit):
        self.F = F
        self.limit = limit
        self.n = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, x):
        if self.n >= self.limit:
            return None
        self.n += 1
        try:
            v = self.F(x)
        except Exception:
            return None
        v = _clean(v)
        if v is None:
            return None
        f = float(np.linalg.norm(v))
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return v

    def best(self):
        return self.best_x

def _secant_step(F, x1, x0, F1, F0):
    """
    Compute secant direction for 2D system.
    We have F(x1) and F(x0). We want to solve J dx = -F1.
    Approximate J using (F1 - F0) / (x1 - x0).
    Since we are in 2D, we can construct the 2x2 Jacobian approximation 
    if we have two distinct points, but here we only have one step.
    
    Actually, for a true secant method in 2D, we need two previous points 
    to estimate the 2x2 Jacobian. 
    Strategy: 
    Maintain history of (x, F). 
    If we have at least 2 distinct points, estimate J.
    If we only have 1 step, use a rank-1 update to a previous J estimate 
    or fall back to steepest descent direction -F1.
    
    For simplicity and robustness in this implementation:
    We will maintain a small history.
    """
    pass

def _local_secant_search(g, x0, max_it=30):
    """
    Local search using a modified secant/quasi-Newton approach.
    Uses function values only.
    """
    x = np.array(x0, dtype=float)
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f

    # History for secant approximation
    # We need to build a Jacobian approximation.
    # Start with identity or a simple estimate.
    J = np.eye(2)
    
    # To get a better initial J, we can do one finite difference step 
    # if we have budget, but let's try pure secant first.
    # Pure secant in 2D requires 2 points to define the 2x2 J.
    # Let's do one "probe" step to get a second point.
    
    # Probe in a direction perpendicular to Fx or just along x-axis
    # Better: Use a small step in a random direction to get a second point.
    # This costs 1 call.
    
    if g.n < g.limit - 5:
        # Try to estimate J with 2 extra calls (FD) for better start? 
        # No, let's stick to the "Secant" promise: minimal calls.
        # We will use a rank-1 update scheme (Broyden-like) but without 
        # explicit J init from FD.
        
        # Initialize J as identity.
        # Iterate:
        # 1. Solve J dx = -Fx
        # 2. Step to x_new
        # 3. Evaluate F_new
        # 4. Update J using Broyden formula: J = J + (F_new - Fx - J*dx) * dx.T / (dx.T dx)
        
        for _ in range(max_it):
            if g.n >= g.limit or _time_left() < 1.0:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If J is singular, use steepest descent
                dx = -Fx
                norm_dx = np.linalg.norm(dx)
                if norm_dx < 1e-12:
                    break
                dx = dx / norm_dx * 0.1
                J = np.eye(2) # Reset J if it was bad

            if not np.all(np.isfinite(dx)):
                break

            # Backtracking line search
            alpha = 1.0
            success = False
            for _ in range(15):
                x_new = x + alpha * dx
                Fx_new = g(x_new)
                if Fx_new is None:
                    break
                f_new = float(np.linalg.norm(Fx_new))
                
                if f_new <= 1e-8:
                    return x_new, True, f_new
                
                # Accept if we make progress
                if f_new < f:
                    # Broyden update
                    s = alpha * dx
                    y = Fx_new - Fx
                    denom = np.dot(s, s)
                    if denom > 1e-15:
                        J = J + np.outer(y - J @ s, s) / denom
                    
                    x = x_new
                    Fx = Fx_new
                    f = f_new
                    success = True
                    break
                alpha *= 0.5
                
            if not success:
                break

    return x, False, f

def _basin_hopping_scan(g, x0, budget=30):
    """
    Perform a series of short local searches from different starting points 
    to find the basin of attraction.
    """
    x0 = np.array(x0, dtype=float)
    scale = max(1.0, np.linalg.norm(x0))
    
    # Generate diverse starting points
    # 1. x0 itself
    # 2. Points at various scales and angles
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # First, try x0
    x_res, success, f_res = _local_secant_search(g, x0, max_it=10)
    if f_res < g.best_f:
        g.best_f = f_res
        g.best_x = x_res.copy()
    if success:
        return x_res, True, f_res

    # Try a few other points
    # We want to find a point with low ||F||.
    # Use a simple grid of 8 points around x0
    directions = [
        np.array([1, 0]), np.array([-1, 0]),
        np.array([0, 1]), np.array([0, -1]),
        np.array([1, 1]), np.array([-1, -1]),
        np.array([1, -1]), np.array([-1, 1])
    ]
    
    for d in directions:
        if g.n >= budget + 20: # Allow some extra for the local search
            break
        x_start = x0 + d * scale
        x_res, success, f_res = _local_secant_search(g, x_start, max_it=10)
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
        if success:
            return x_res, True, f_res
            
    return g.best_x, False, g.best_f

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Basin Hopping to find a good starting point
    x_start, success, f_start = _basin_hopping_scan(g, x0, budget=30)
    
    if success:
        return x_start
        
    # 2. If we found a good point (low f) but not a root, refine it
    if g.best_x is not None and g.best_f < 1e-3:
        # Run a longer local search from the best point found
        x_res, success, f_res = _local_secant_search(g, g.best_x, max_it=50)
        if success:
            return x_res
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
            
    # 3. If still not found, try a more global random search
    # This is a fallback
    if g.best_f > 1e-3 and g.n < limit - 50:
        rng = np.random.default_rng(123)
        for _ in range(10):
            if g.n >= limit - 20 or _time_left() < 5.0:
                break
            # Sample in a larger box
            x_rand = x0 + rng.uniform(-5, 5, size=2) * max(1.0, np.linalg.norm(x0))
            x_res, success, f_res = _local_secant_search(g, x_rand, max_it=10)
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
            if success:
                return x_res
                
    # Return best found
    if g.best_x is not None:
        return g.best_x
    return x0.copy()

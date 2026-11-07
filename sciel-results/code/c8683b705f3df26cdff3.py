"""
Root finder for 3 fixed 2-D systems: Secant-Step Hybrid with Periodic Re-anchoring.

Strategy:
1. Global Phase: Perform a coarse scan (grid + random) to find a point with
   low ||F(x)||. This identifies the basin of attraction.
2. Local Phase: Use a Secant-Step Hybrid method:
   - Initialize the Jacobian via a two-point secant difference (costing 2 calls)
   - Alternate between standard Broyden rank-1 updates and a full 3-call
     finite-difference Jacobian recomputation every k iterations to correct
     accumulated approximation error (re-anchoring)
   - This prevents the quasi-Newton matrix from drifting into ill-conditioning
3. Polish Phase: When close enough (||F|| < 1e-4), switch to a damped Newton
   method with central differences for rapid, guaranteed convergence to 1e-8.
4. Fallback: If the primary local search fails, try a few other top global
   candidates.

This approach trades some initial global exploration calls for a much more
efficient local convergence, aiming to stay well under the 500 call limit
while robustly handling systems with varying basin structures.
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

def _secant_initialize(g, x0):
    """
    Initialize the Jacobian via a two-point secant difference.
    Returns (J, x, Fx, f, success)
    """
    x = np.array(x0, dtype=float)
    Fx = g(x)
    if Fx is None:
        return None, x, None, np.inf, False
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return None, x, Fx, f, True

    # Choose a perturbation direction
    scale = max(1.0, np.linalg.norm(x))
    h = 1e-4 * scale
    
    # Perturb along the direction of Fx (or a default direction if Fx is small)
    if f > 1e-10:
        d = Fx / f
    else:
        d = np.array([1.0, 0.0])
    
    x1 = x + h * d
    Fx1 = g(x1)
    if Fx1 is None:
        return None, x, Fx, f, False
    
    # Compute secant Jacobian: J * (x1 - x) = Fx1 - Fx
    s = x1 - x
    y = Fx1 - Fx
    
    # Solve J * s = y for J (2x2 system)
    # J * s = y => J = y * s^T / (s^T * s) is not quite right for 2x2
    # We need to solve for J such that J @ s = y
    # This is underdetermined (2 equations, 4 unknowns)
    # We use the minimum-norm solution: J = y @ inv(s @ s^T) @ s^T ... 
    # Actually, let's use a different approach: 
    # We have J @ s = y. We can pick J = y @ s^T / dot(s,s) as a rank-1 approximation
    # But this might be singular. Let's use a more robust approach:
    # Solve for J using two secant directions
    
    # Better: use two perturbations to get a full Jacobian estimate
    # But that costs 2 more calls. Instead, let's use the rank-1 secant
    # and regularize it.
    
    denom = np.dot(s, s)
    if denom < 1e-15:
        return None, x, Fx, f, False
    
    # Rank-1 secant: J = y @ s^T / dot(s, s)
    J = np.outer(y, s) / denom
    
    # Regularize: add a small identity to ensure invertibility
    J = J + 1e-6 * np.eye(2)
    
    return J, x, Fx, f, False

def _broyden_with_reanchor(g, x0, max_it=40, reanchor_every=3):
    """
    Broyden's method with periodic re-anchoring via finite-difference Jacobian.
    Returns (x, success, f_val)
    """
    x = np.array(x0, dtype=float)
    
    # Initialize Jacobian via secant
    J, x, Fx, f, found = _secant_initialize(g, x)
    if found:
        return x, True, f
    if J is None:
        return x, False, f
    
    for it in range(max_it):
        if g.n >= g.limit or _time_left() < 1.0:
            break
        
        # Switch to Newton polish if we are close enough
        if f < 1e-4:
            x_pol, success, f_pol = _newton_polish(g, x, Fx)
            if success:
                return x_pol, True, f_pol
            # If polish fails, continue with Broyden from the polished x
            x = x_pol
            Fx = g(x)
            if Fx is None:
                break
            f = float(np.linalg.norm(Fx))
            if f <= 1e-8:
                return x, True, f
            continue
        
        # Re-anchor every reanchor_every iterations
        if (it > 0) and (it % reanchor_every == 0):
            # Recompute Jacobian via finite differences (3 calls)
            h = 1e-6 * max(1.0, np.linalg.norm(x))
            J_new = np.zeros((2, 2))
            valid = True
            for j in range(2):
                xp = x.copy()
                xm = x.copy()
                xp[j] += h
                xm[j] -= h
                Fp = g(xp)
                Fm = g(xm)
                if Fp is None or Fm is None:
                    valid = False
                    break
                J_new[:, j] = (Fp - Fm) / (2 * h)
            if valid:
                J = J_new
                # Update Fx if needed (it should be current)
            
            # If re-anchoring failed, fall back to Broyden update
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            J = np.eye(2)
            dx = -Fx * 0.1
            
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
            
            # Accept if we make progress (Armijo-like condition simplified)
            if f_new < f:
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

def _newton_polish(g, x0, Fx0, max_it=10):
    """
    Damped Newton with central differences for final convergence.
    Uses 3 calls per iteration (F(x), F(x+h), F(x-h)).
    """
    x = np.array(x0, dtype=float)
    Fx = Fx0.copy()
    f = float(np.linalg.norm(Fx))
    
    if f <= 1e-8:
        return x, True, f

    h = 1e-6
    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 0.5:
            break
            
        # Compute Jacobian using central differences
        J = np.zeros((2, 2))
        valid = True
        for j in range(2):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            
            Fp = g(xp)
            Fm = g(xm)
            if Fp is None or Fm is None:
                valid = False
                break
            J[:, j] = (Fp - Fm) / (2 * h)
        
        if not valid:
            break
            
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
            
        if not np.all(np.isfinite(dx)):
            break
            
        # Damped step
        alpha = 1.0
        success = False
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            
            if f_new <= 1e-8:
                return x_new, True, f_new
                
            if f_new < f:
                x = x_new
                Fx = Fx_new
                f = f_new
                success = True
                break
            alpha *= 0.5
            
        if not success:
            break
            
    return x, False, f

def _global_scan(g, x0, budget=50):
    """
    Fallback global search.
    Returns list of (f_val, x) sorted by f_val.
    """
    x0 = np.array(x0, dtype=float)
    scale = max(1.0, np.linalg.norm(x0))
    candidates = []
    
    # Simple grid around x0
    offsets = [
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [2, 0], [-2, 0], [0, 2], [0, -2],
        [1, 1], [-1, -1], [1, -1], [-1, 1],
        [3, 0], [-3, 0], [0, 3], [0, -3],
        [2, 2], [-2, -2], [2, -2], [-2, 2],
        [4, 0], [-4, 0], [0, 4], [0, -4]
    ]
    
    for off in offsets:
        if g.n >= budget:
            break
        x = x0 + np.array(off) * scale
        fx = g(x)
        if fx is not None:
            candidates.append((float(np.linalg.norm(fx)), x))
            
    # Random points in a wider box
    rng = np.random.default_rng(42)
    while g.n < budget:
        x = x0 + rng.uniform(-3 * scale, 3 * scale, size=2)
        fx = g(x)
        if fx is not None:
            candidates.append((float(np.linalg.norm(fx)), x))
            
    candidates.sort(key=lambda t: t[0])
    return candidates

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Global Scan to find good starting points
    candidates = _global_scan(g, x0, budget=40)
    
    if g.best_f <= 1e-8:
        return g.best_x

    # 2. Try Broyden with re-anchoring from top candidates
    # Ensure x0 is in the list if it wasn't scanned
    top_candidates = [c[1] for c in candidates[:5]]
    if not any(np.allclose(c, x0) for c in top_candidates):
        top_candidates.append(x0)
        
    # Deduplicate
    unique_candidates = []
    for c in top_candidates:
        if not any(np.allclose(c, uc) for uc in unique_candidates):
            unique_candidates.append(c)
            
    for x_start in unique_candidates:
        if g.n >= limit - 30 or _time_left() < 5.0:
            break
            
        x_res, success, f_res = _broyden_with_reanchor(g, x_start, max_it=40, reanchor_every=3)
        x_res = _clean(x_res)
        
        if x_res is not None:
            # Check if we found a root
            if f_res <= 1e-8:
                return x_res
                
            # Update best
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
                
            # If we got very close but didn't hit 1e-8, try a quick Newton polish
            if f_res < 1e-3 and g.n < limit - 10 and _time_left() > 2.0:
                Fx_res = g(x_res)
                if Fx_res is not None:
                    x_pol, success_pol, f_pol = _newton_polish(g, x_res, Fx_res)
                    x_pol = _clean(x_pol)
                    if x_pol is not None and f_pol <= 1e-8:
                        return x_pol
                    if x_pol is not None and f_pol < g.best_f:
                        g.best_f = f_pol
                        g.best_x = x_pol.copy()

    # Return best found
    if g.best_x is not None:
        return g.best_x
    return x0.copy()

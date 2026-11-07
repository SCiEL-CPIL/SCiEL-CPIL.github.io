"""
Root finder for 3 fixed 2-D systems: Homotopy Path Following Strategy.

Strategy:
1. Problem Framing: Instead of searching for a basin of attraction, we construct
   a homotopy H(x, t) = (1-t)*x + t*F(x).
   At t=0, H(x,0)=x, so the root is x=0 (known).
   At t=1, H(x,1)=F(x), so the root is the target.
   We track the solution x(t) from t=0 to t=1.
2. Local Solver: Use a damped Newton method with finite-difference Jacobian
   for the homotopy system. This is robust and handles the changing structure
   of the system as t increases.
3. Adaptive Step Size: Start with small t steps (0.01) and increase them
   (up to 0.1) if Newton converges quickly. If Newton fails, reduce step size.
4. Safety: If the path breaks (Newton fails even with small steps), fallback
   to a global grid search + Broyden (similar to previous strategy) but only
   if budget allows.

This approach is structurally different because it uses the topology of the
solution set (connectedness via homotopy) rather than spatial proximity.
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

def _newton_homotopy(g, x_start, t, max_it=20):
    """
    Solve H(x, t) = 0 for x, starting from x_start.
    H(x, t) = (1-t)*x + t*F(x)
    J_H(x, t) = (1-t)*I + t*J_F(x)
    
    Returns (x_sol, success, calls_used)
    """
    x = np.array(x_start, dtype=float)
    calls = 0
    
    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 0.5:
            return x, False, calls
        
        # Evaluate F(x)
        Fx = g(x)
        if Fx is None:
            return x, False, calls
        calls += 1
        
        # H(x) = (1-t)*x + t*Fx
        H = (1 - t) * x + t * Fx
        f_H = float(np.linalg.norm(H))
        
        if f_H <= 1e-12: # Converged
            return x, True, calls
        
        # Compute Jacobian of H: J_H = (1-t)*I + t*J_F
        # Use central difference for J_F to improve accuracy
        h = 1e-5
        J_F = np.zeros((2, 2))
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
            calls += 2
            J_F[:, j] = (Fp - Fm) / (2 * h)
        
        if not valid:
            return x, False, calls
            
        J_H = (1 - t) * np.eye(2) + t * J_F
        
        try:
            dx = np.linalg.solve(J_H, -H)
        except np.linalg.LinAlgError:
            return x, False, calls
            
        if not np.all(np.isfinite(dx)):
            return x, False, calls
            
        # Damped step
        alpha = 1.0
        success = False
        for _ in range(10):
            x_new = x + alpha * dx
            H_new = (1 - t) * x_new + t * g(x_new)
            if H_new is None:
                break
            calls += 1
            f_H_new = float(np.linalg.norm(H_new))
            
            if f_H_new < f_H * 0.9: # Sufficient decrease
                x = x_new
                success = True
                break
            alpha *= 0.5
            
        if not success:
            return x, False, calls
            
    return x, False, calls

def _homotopy_search(g, x0):
    """
    Perform homotopy path following from t=0 to t=1.
    """
    x = np.zeros(2) # Root of H(x,0)=x is 0
    t = 0.0
    step = 0.01
    
    while t < 1.0:
        if g.n >= g.limit or _time_left() < 2.0:
            break
            
        # Try to advance to t + step
        t_new = min(t + step, 1.0)
        
        x_sol, success, _ = _newton_homotopy(g, x, t_new, max_it=15)
        
        if success:
            x = x_sol
            t = t_new
            # Increase step size if we moved easily
            step = min(step * 1.5, 0.1)
        else:
            # Reduce step size
            step *= 0.5
            if step < 1e-4:
                break
                
    return x

def _global_scan(g, x0, budget=30):
    """
    Fallback global search.
    """
    x0 = np.array(x0, dtype=float)
    scale = max(1.0, np.linalg.norm(x0))
    candidates = []
    
    # Simple grid
    offsets = [
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [2, 0], [-2, 0], [0, 2], [0, -2],
        [1, 1], [-1, -1], [1, -1], [-1, 1]
    ]
    
    for off in offsets:
        if g.n >= budget:
            break
        x = x0 + np.array(off) * scale
        fx = g(x)
        if fx is not None:
            candidates.append((float(np.linalg.norm(fx)), x))
            
    # Random
    rng = np.random.default_rng(42)
    while g.n < budget:
        x = x0 + rng.uniform(-2 * scale, 2 * scale, size=2)
        fx = g(x)
        if fx is not None:
            candidates.append((float(np.linalg.norm(fx)), x))
            
    candidates.sort(key=lambda t: t[0])
    return candidates

def _broyden_local(g, x0, max_it=30):
    """
    Fallback Broyden solver.
    """
    x = np.array(x0, dtype=float)
    J = np.eye(2)
    
    Fx = g(x)
    if Fx is None:
        return x
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x

    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 0.5:
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            J = np.eye(2)
            dx = -Fx
            
        if not np.all(np.isfinite(dx)):
            break
            
        alpha = 1.0
        success = False
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            
            if f_new <= 1e-8:
                return x_new
            
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
            
    return x

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Try Homotopy Path Following
    x_hom = _homotopy_search(g, x0)
    x_hom = _clean(x_hom)
    
    if x_hom is not None:
        fx = g(x_hom)
        if fx is not None:
            f_val = float(np.linalg.norm(fx))
            if f_val <= 1e-8:
                return x_hom
            
    # 2. Fallback: Global Scan + Broyden
    # Only if we have budget left
    if g.n < limit - 50 and _time_left() > 5.0:
        candidates = _global_scan(g, x0, budget=40)
        
        if g.best_f <= 1e-8:
            return g.best_x
            
        top_candidates = [c[1] for c in candidates[:3]]
        if not any(np.allclose(c, x0) for c in top_candidates):
            top_candidates.append(x0)
            
        for x_start in top_candidates:
            if g.n >= limit - 20 or _time_left() < 2.0:
                break
            x_res = _broyden_local(g, x_start, max_it=30)
            x_res = _clean(x_res)
            if x_res is not None:
                fx = g(x_res)
                if fx is not None:
                    f_val = float(np.linalg.norm(fx))
                    if f_val < g.best_f:
                        g.best_f = f_val
                        g.best_x = x_res
                    if f_val <= 1e-8:
                        return x_res
                        
    # Return best found
    if g.best_x is not None:
        return g.best_x
    return x0.copy()

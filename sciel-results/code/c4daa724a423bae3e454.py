"""
Root finder for 3 fixed 2-D systems: Global Direct Search with Newton Polish.

Strategy:
1. Use Nelder-Mead simplex method to minimize the merit function ||F(x)||^2.
   This is a derivative-free global search that is robust to initial conditions
   and does not require Jacobian approximation, avoiding the pitfalls of
   quasi-Newton methods on difficult landscapes.
2. Once Nelder-Mead converges or exhausts its budget, use a final Newton step
   with a finite-difference Jacobian to polish the solution to the required
   1e-8 tolerance.
3. If the initial Nelder-Mead run fails to find a root, perform a few restarts
   with different initial simplex shapes to escape local minima of the merit
   function.

This approach trades the efficiency of quasi-Newton methods for the robustness
of direct search, which is often superior for 2-D problems with unknown
landscape characteristics.
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

def _merit(g, x):
    """Compute merit function ||F(x)||^2, returning np.inf if invalid."""
    v = g(x)
    if v is None:
        return np.inf
    return float(np.dot(v, v))

def _nelder_mead(g, x0, max_iter=50, tol=1e-12):
    """
    Nelder-Mead simplex method to minimize ||F(x)||^2.
    Returns (x_best, f_best, success).
    """
    # Initialize simplex: x0, x0 + h*e1, x0 + h*e2
    h = 0.1 * max(1.0, np.linalg.norm(x0))
    x0 = np.array(x0, dtype=float)
    simplex = [
        x0.copy(),
        x0 + np.array([h, 0.0]),
        x0 + np.array([0.0, h])
    ]
    
    f_vals = []
    for x in simplex:
        f = _merit(g, x)
        f_vals.append(f)
        if f < g.best_f:
            g.best_f = f
            g.best_x = x.copy()
    
    # Sort simplex by f value
    indices = np.argsort(f_vals)
    simplex = [simplex[i] for i in indices]
    f_vals = [f_vals[i] for i in indices]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    for _ in range(max_iter):
        if g.n >= g.limit - 10 or _time_left() < 1.0:
            break
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
        
        # Reflection
        x_worst = simplex[-1]
        x_centroid = np.mean(simplex[:-1], axis=0)
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = _merit(g, x_r)
        if f_r < g.best_f:
            g.best_f = f_r
            g.best_x = x_r.copy()
        
        if f_vals[0] <= f_r < f_vals[-2]:
            # Accept reflection
            simplex[-1] = x_r
            f_vals[-1] = f_r
        elif f_r < f_vals[0]:
            # Expansion
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = _merit(g, x_e)
            if f_e < g.best_f:
                g.best_f = f_e
                g.best_x = x_e.copy()
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        else:
            # Contraction
            if f_r < f_vals[-1]:
                # Outside contraction
                x_c = x_centroid + 0.5 * (x_r - x_centroid)
            else:
                # Inside contraction
                x_c = x_centroid - 0.5 * (x_worst - x_centroid)
            f_c = _merit(g, x_c)
            if f_c < g.best_f:
                g.best_f = f_c
                g.best_x = x_c.copy()
            
            if f_c < min(f_r, f_vals[-1]):
                simplex[-1] = x_c
                f_vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, 3):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_vals[i] = _merit(g, simplex[i])
                    if f_vals[i] < g.best_f:
                        g.best_f = f_vals[i]
                        g.best_x = simplex[i].copy()
        
        # Re-sort simplex
        indices = np.argsort(f_vals)
        simplex = [simplex[i] for i in indices]
        f_vals = [f_vals[i] for i in indices]
        
        if f_vals[0] <= 1e-16:
            break
    
    return simplex[0], f_vals[0], f_vals[0] <= 1e-16

def _newton_polish(g, x0, max_it=20):
    """
    Newton's method with finite-difference Jacobian to polish the solution.
    Returns (x, f, success).
    """
    x = np.array(x0, dtype=float)
    Fx = g(x)
    if Fx is None:
        return x, np.inf, False
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, f, True
    
    for _ in range(max_it):
        if g.n >= g.limit - 5 or _time_left() < 1.0:
            break
        
        # Compute FD Jacobian (3 calls)
        h = 1e-6
        Fx0 = Fx
        J = np.zeros((2, 2))
        valid = True
        for i in range(2):
            xp = x.copy()
            xp[i] += h
            Fp = g(xp)
            if Fp is None:
                valid = False
                break
            J[:, i] = (Fp - Fx0) / h
        
        if not valid:
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Backtracking line search
        alpha = 1.0
        success = False
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            if f_new <= 1e-8:
                return x_new, f_new, True
            if f_new < f:
                x = x_new
                Fx = Fx_new
                f = f_new
                success = True
                break
            alpha *= 0.5
        
        if not success:
            break
    
    return x, f, f <= 1e-8

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Initial Nelder-Mead run from x0
    x_nm, f_nm, success_nm = _nelder_mead(g, x0, max_iter=30)
    
    if success_nm:
        # Polish with Newton
        x_newton, f_newton, success_newton = _newton_polish(g, x_nm)
        if success_newton:
            return x_newton
        if f_newton < f_nm:
            x_nm = x_newton
            f_nm = f_newton
    
    # 2. If not successful, try a few restarts with different initial simplexes
    if not success_nm and g.n < limit - 50:
        rng = np.random.default_rng(42)
        for _ in range(3):
            if g.n >= limit - 20 or _time_left() < 5.0:
                break
            # Random restart near x0
            x_restart = x0 + rng.uniform(-1, 1, size=2) * max(1.0, np.linalg.norm(x0))
            x_nm, f_nm, success_nm = _nelder_mead(g, x_restart, max_iter=20)
            if success_nm:
                x_newton, f_newton, success_newton = _newton_polish(g, x_nm)
                if success_newton:
                    return x_newton
                if f_newton < f_nm:
                    x_nm = x_newton
                    f_nm = f_newton
    
    # 3. Final polish attempt from the best point found
    if g.best_x is not None:
        x_final, f_final, success_final = _newton_polish(g, g.best_x, max_it=10)
        if success_final:
            return x_final
        if f_final < g.best_f:
            g.best_f = f_final
            g.best_x = x_final.copy()
    
    # Return best found
    if g.best_x is not None:
        return g.best_x
    return x0.copy()

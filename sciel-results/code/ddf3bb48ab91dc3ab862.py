"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Strategy: Broyden's Method with Global Line Search and Restarts
    # Key difference from Newton:
    # 1. Broyden's method maintains a Jacobian approximation J and updates it
    #    using only 1 F-call per iteration (plus the initial setup).
    #    Newton requires ~3-5 F-calls per iteration for finite-difference Jacobian.
    # 2. This allows many more iterations within the call budget, potentially
    #    escaping local minima or singularities that stall Newton.
    # 3. We use Broyden's "good" (inverse) update for stability.
    # 4. If Broyden stalls, we restart with a fresh Jacobian from a nearby point.
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def norm(vec):
        return np.linalg.norm(vec)
    
    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False
    
    def finite_diff_jac(F, x, eps=1e-6):
        """Compute Jacobian using forward differences (3 calls total: F(x) + 2 perturbed)."""
        n = len(x)
        J = np.zeros((n, n))
        Fx = F(x)
        for j in range(n):
            e = np.zeros(n)
            e[j] = eps
            try:
                J[:, j] = (F(x + e) - Fx) / eps
            except Exception:
                return None
        if not np.all(np.isfinite(J)):
            return None
        return J, Fx
    
    def broyden(F, x0, max_iter=100, tol=1e-12):
        """
        Broyden's method with line search.
        Returns (best_x, best_norm, converged)
        """
        x = np.array(x0, dtype=float)
        
        # Initial Jacobian setup
        try:
            J, Fx = finite_diff_jac(F, x)
        except Exception:
            return x, np.inf, False
        
        if J is None:
            return x, np.inf, False
        
        best_x = x.copy()
        best_norm = norm(Fx)
        
        for i in range(max_iter):
            if time_left() < 0.5:
                break
            
            current_norm = best_norm
            if current_norm <= tol:
                return x, current_norm, True
            
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Try pseudo-inverse or restart
                break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Line search with Broyden update
            step_found = False
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
                x_new = x + lam * dx
                try:
                    F_new = F(x_new)
                except Exception:
                    continue
                
                if not np.all(np.isfinite(F_new)):
                    continue
                
                new_norm = norm(F_new)
                
                # Accept if norm decreases
                if new_norm < current_norm:
                    # Broyden's "good" update: J_new = J + (F_new - Fx - J*dx) * (dx^T) / (dx^T * dx)
                    # More stable form: J_new = J + (deltaF - J*dx) * dx^T / (dx^T*dx)
                    delta_F = F_new - Fx
                    Jdx = J @ dx
                    correction = (delta_F - Jdx)[:, None] * dx[None, :]
                    denom = np.dot(dx, dx)
                    if denom > 1e-15:
                        J = J + correction / denom
                    
                    x = x_new
                    Fx = F_new
                    current_norm = new_norm
                    
                    if current_norm < best_norm:
                        best_norm = current_norm
                        best_x = x.copy()
                    
                    step_found = True
                    break
            
            if not step_found:
                break
            
            # Convergence check on step size
            if norm(dx) < 1e-10:
                break
        
        return best_x, best_norm, False
    
    # Phase 1: Broyden from x0
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    x_best, norm_best, converged = broyden(F, x0)
    
    if norm_best <= 1e-8:
        return x_best
    
    # Phase 2: Restarts with Broyden
    # Generate candidates around x0 and zero
    scale = max(1.0, norm(x0))
    
    candidates = []
    candidates.append(np.array([0.0, 0.0]))
    candidates.append(x0 * 0.5)
    candidates.append(x0 * 2.0)
    candidates.append(x0 * 0.1)
    candidates.append(x0 * 5.0)
    
    rng = np.random.default_rng(42)
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0, scale * 5.0]:
        for _ in range(3):
            u = rng.normal(size=2)
            u = u / norm(u)
            p = x0 + r * u
            candidates.append(p)
    
    for c in candidates:
        if time_left() < 1.0:
            break
        
        if not np.all(np.isfinite(c)):
            continue
        
        x_try, norm_try, conv = broyden(F, c)
        
        if norm_try <= 1e-8:
            return x_try
        
        if norm_try < norm_best:
            x_best = x_try
            norm_best = norm_try
    
    return x_best
# EVOLVE-BLOCK-END

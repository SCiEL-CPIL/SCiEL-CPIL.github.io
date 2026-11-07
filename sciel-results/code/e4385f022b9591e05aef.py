"""Root finder for 3 fixed 2-D systems: Broyden hybrid with safeguards."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    - Broyden's method (rank-1) for fast convergence after initial Jacobian estimate
    - Periodic full finite-difference Jacobian recomputation to correct drift
    - Damped steps to prevent divergence
    - Multiple restart strategies if stuck
    - Explicit time and call budget guards
    """
    start_time = time.time()
    time_limit = 110.0  # safety margin before 120s kill
    call_count = 0
    max_calls = 450  # stay under 500 per system
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    def safe_F(x):
        nonlocal call_count
        if call_count >= max_calls:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x, dtype=float))
            result = np.asarray(result, dtype=float)
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None
    
    def fd_jacobian(x, Fx):
        """Compute Jacobian via forward differences (3 calls: Fx + 2 perturbations)."""
        nonlocal call_count
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def newton_step(x, Fx, J):
        """Compute Newton step with damping."""
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            return None
        # Damp if step is too large
        norm_step = np.linalg.norm(step)
        norm_Fx = np.linalg.norm(Fx)
        if norm_step > 10 * max(1.0, norm_Fx):
            step = step / (norm_step / (10 * max(1.0, norm_Fx)))
        return x - step
    
    def broyden_update(J, Fx, x, x_new, Fx_new):
        """Rank-1 Broyden update."""
        s = x_new - x
        y = Fx_new - Fx
        denom = np.dot(y, s)
        if abs(denom) < 1e-15:
            return J
        return J + np.outer(y - J @ s, s) / denom
    
    def try_solve(x0_start, label=""):
        """Attempt to find a root starting from x0_start using Broyden hybrid."""
        nonlocal call_count, time_ok
        
        if not time_ok() or call_count >= max_calls:
            return None, None
        
        x = np.array(x0_start, dtype=float)
        Fx = safe_F(x)
        if Fx is None:
            return None, None
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x, norm_Fx
        
        # Initial Jacobian estimate via finite differences
        J = fd_jacobian(x, Fx)
        if J is None:
            return None, None
        
        best_x = x.copy()
        best_norm = norm_Fx
        
        for it in range(50):
            if not time_ok() or call_count >= max_calls:
                break
            
            # Every 5 iterations, recompute full Jacobian to correct drift
            if it > 0 and it % 5 == 0:
                J = fd_jacobian(x, Fx)
                if J is None:
                    break
            
            # Compute step
            step = newton_step(x, Fx, J)
            if step is None:
                # Fallback: try steepest descent direction
                step = -Fx * 0.1
                norm_step = np.linalg.norm(step)
                if norm_step > 1.0:
                    step = step / norm_step
            
            x_new = x + step
            Fx_new = safe_F(x_new)
            if Fx_new is None:
                break
            
            norm_Fx_new = np.linalg.norm(Fx_new)
            
            # Accept if improvement, or if we're making progress
            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new
            
            if norm_Fx_new <= 1e-8:
                return x_new, norm_Fx_new
            
            # Broyden update
            J = broyden_update(J, Fx, x, x_new, Fx_new)
            if J is None:
                break
            
            x, Fx = x_new, Fx_new
        
        return best_x, best_norm
    
    # Try initial point
    x_best, norm_best = try_solve(x0)
    if norm_best is not None and norm_best <= 1e-8:
        return x_best
    
    # If not converged, try several restart strategies
    restarts = []
    x0_arr = np.array(x0, dtype=float)
    
    # Small perturbations
    for scale in [0.5, 2.0, -1.0, 0.1, -0.5]:
        restarts.append(x0_arr * scale)
    
    # Random perturbations around x0
    np.random.seed(42)
    for _ in range(3):
        restarts.append(x0_arr + np.random.randn(2) * 0.5)
    
    # Origin
    restarts.append(np.zeros(2))
    
    for i, x_start in enumerate(restarts):
        if not time_ok() or call_count >= max_calls:
            break
        
        x_candidate, norm_candidate = try_solve(x_start)
        
        if norm_candidate is not None:
            if norm_candidate <= 1e-8:
                return x_candidate
            if norm_best is None or norm_candidate < norm_best:
                x_best, norm_best = x_candidate, norm_candidate
    
    # Return best found
    if x_best is not None:
        return x_best
    return np.array(x0, dtype=float)

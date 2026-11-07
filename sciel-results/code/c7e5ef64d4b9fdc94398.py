"""Root finder for 3 fixed 2D systems using homotopy/continuation method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses homotopy/continuation method to trace a path from a known solvable anchor
    to the target system, then refines with Newton.
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def time_up():
        return time.time() > deadline

    def norm_F(x_val):
        try:
            Fx = F(x_val)
            if not np.all(np.isfinite(Fx)):
                return np.inf
            return np.linalg.norm(Fx)
        except Exception:
            return np.inf

    # Check if x0 is already a root
    Fx = F(x)
    if np.all(np.isfinite(Fx)):
        if np.linalg.norm(Fx) <= 1e-8:
            return x

    # Strategy: Use homotopy/continuation from a simple anchor system
    # Anchor: G(x) = x - x_anchor = 0, so root is x_anchor
    # We'll trace from anchor to target by blending: H_t(x) = (1-t)*G(x) + t*F(x)
    # Start at t=0 (easy root at x_anchor), increase t to 1 (target system)
    
    # Choose anchor points to try - use x0 and some perturbations
    anchor_candidates = [x.copy(), x + 1.0, x - 1.0, x + 0.1, x - 0.1]
    
    best_x = x.copy()
    best_norm = norm_F(x)
    
    for anchor in anchor_candidates:
        if time_up():
            break
        
        # Homotopy: H_t(x) = (1-t)*(x - anchor) + t*F(x)
        # At t=0: H_0(x) = x - anchor, root at x = anchor
        # At t=1: H_1(x) = F(x), root at target
        
        x_curr = anchor.copy()
        t_curr = 0.0
        dt = 0.1  # Initial step size in t
        
        # Trace the path from t=0 to t=1
        for t_step in range(20):
            if time_up():
                break
            
            t_target = min(t_curr + dt, 1.0)
            if t_target >= 1.0:
                t_target = 1.0
            
            # Solve H_t(x) = 0 using Newton with finite differences
            x_new = x_curr.copy()
            converged = False
            
            for newton_iter in range(15):
                if time_up():
                    break
                
                # Evaluate H_t and its Jacobian
                def H_t(x_val):
                    return (1 - t_target) * (x_val - anchor) + t_target * F(x_val)
                
                Hx = H_t(x_new)
                if not np.all(np.isfinite(Hx)):
                    break
                
                norm_Hx = np.linalg.norm(Hx)
                if norm_Hx <= 1e-10:
                    converged = True
                    break
                
                # Finite difference Jacobian
                h_fd = 1e-6
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h_fd
                    J[:, j] = (H_t(x_new + e) - H_t(x_new - e)) / (2 * h_fd)
                
                try:
                    d = np.linalg.solve(J, -Hx)
                except np.linalg.LinAlgError:
                    break
                
                if np.linalg.norm(d) < 1e-12:
                    converged = True
                    break
                
                # Line search
                alpha = 1.0
                step_ok = False
                for ls in range(10):
                    x_trial = x_new + alpha * d
                    H_trial = H_t(x_trial)
                    if not np.all(np.isfinite(H_trial)):
                        alpha *= 0.5
                        continue
                    
                    if np.linalg.norm(H_trial) < norm_Hx:
                        x_new = x_trial
                        step_ok = True
                        break
                    alpha *= 0.5
                
                if not step_ok:
                    break
            
            if converged:
                x_curr = x_new
                t_curr = t_target
                if t_curr >= 1.0:
                    # Reached target, check if it's a good root
                    Fx_check = F(x_curr)
                    if np.all(np.isfinite(Fx_check)):
                        norm_check = np.linalg.norm(Fx_check)
                        if norm_check <= 1e-8:
                            return x_curr
                        if norm_check < best_norm:
                            best_norm = norm_check
                            best_x = x_curr.copy()
                    break
            else:
                # Failed to converge at this t, try smaller step
                dt = max(dt * 0.5, 0.01)
                continue
        
        # If we made progress, try to refine with direct Newton on F
        if t_curr > 0.5:
            x_start = x_curr.copy()
            Fx_start = F(x_start)
            if np.all(np.isfinite(Fx_start)) and np.linalg.norm(Fx_start) < best_norm:
                # Newton refinement
                h = 1e-5
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    J[:, j] = (F(x_start + e) - F(x_start - e)) / (2 * h)
                
                x_ref = x_start.copy()
                Fx_ref = Fx_start.copy()
                for n_iter in range(20):
                    if time_up():
                        break
                    
                    try:
                        d = np.linalg.solve(J, -Fx_ref)
                    except np.linalg.LinAlgError:
                        break
                    
                    if np.linalg.norm(d) < 1e-12:
                        break
                    
                    x_ref = x_ref + d
                    Fx_ref = F(x_ref)
                    if not np.all(np.isfinite(Fx_ref)):
                        break
                    
                    norm_ref = np.linalg.norm(Fx_ref)
                    if norm_ref <= 1e-8:
                        return x_ref
                    if norm_ref < best_norm:
                        best_norm = norm_ref
                        best_x = x_ref.copy()
                    
                    # Update Jacobian
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h
                        J[:, j] = (F(x_ref + e) - F(x_ref - e)) / (2 * h)

    # Final fallback: SciPy root from best point
    try:
        res = root(F, best_x, method='hybr', options={'maxfev': 200})
        if np.all(np.isfinite(res.x)) and np.linalg.norm(res.fun) <= 1e-8:
            return res.x
        if np.linalg.norm(res.fun) < best_norm:
            return res.x
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END

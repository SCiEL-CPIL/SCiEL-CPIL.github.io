"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's method with adaptive restarts.
    Broyden's method updates the Jacobian approximation each step, requiring only
    1 function call per iteration after the initial setup (vs 3 for finite differences).
    This is significantly more call-efficient than Newton with finite differences.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x0 = np.array(x0, dtype=float).copy()
    
    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    def check_valid(x):
        return np.all(np.isfinite(x)) and x.shape == (2,)

    # Initial evaluation
    best_x = x0.copy()
    Fx = compute_F(x0)
    if not np.all(np.isfinite(Fx)):
        return x0
    best_norm = np.linalg.norm(Fx)
    
    if best_norm <= 1e-8:
        return best_x

    # Broyden's method with multiple restart strategies
    # We'll try from x0 and a few perturbed points
    
    def broyden_from(start_x, max_iters=100):
        """Run Broyden's method from start_x, return (x, norm_F, num_calls)."""
        x = np.array(start_x, dtype=float).copy()
        Fx = compute_F(x)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf, 1
        norm_F = np.linalg.norm(Fx)
        if norm_F <= 1e-8:
            return x, norm_F, 1
        
        # Initialize Jacobian approximation using finite differences
        # This costs 3 calls (1 for Fx already done, 2 more)
        h = 1e-8 * max(1.0, np.linalg.norm(x))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = compute_F(x + e)
            if not np.all(np.isfinite(Fx_e)):
                # Use identity as fallback
                J = np.eye(2)
                break
            J[:, j] = (Fx_e - Fx) / h
        else:
            # All finite differences succeeded
            pass
        
        # If J is singular or bad, use identity
        if not np.all(np.isfinite(J)) or np.linalg.cond(J) > 1e12:
            J = np.eye(2)
        
        num_calls = 3  # Fx + 2 more
        
        for it in range(max_iters):
            if not time_ok():
                return x, norm_F, num_calls
            
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Use pseudoinverse or break
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            # Damped step with backtracking
            step_size = 1.0
            converged = False
            for _ in range(20):
                x_new = x + step_size * dx
                Fx_new = compute_F(x_new)
                num_calls += 1
                if not np.all(np.isfinite(Fx_new)):
                    step_size *= 0.5
                    continue
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < norm_F:
                    # Update Broyden's Jacobian
                    s = x_new - x
                    y = Fx_new - Fx
                    # Broyden update: J_new = J + (y - J*s) * s^T / (s^T * s)
                    sTs = np.dot(s, s)
                    if sTs > 1e-30:
                        J = J + np.outer(y - J @ s, s) / sTs
                    
                    x = x_new
                    Fx = Fx_new
                    norm_F = norm_new
                    converged = True
                    break
                else:
                    step_size *= 0.5
                    if step_size < 1e-15:
                        break
            
            if not converged:
                break
            
            if norm_F <= 1e-8:
                break
        
        return x, norm_F, num_calls

    # Strategy: Try Broyden from x0, then from perturbed points if needed
    candidates = [x0]
    
    # Add some perturbed starting points
    x_scale = max(1.0, np.linalg.norm(x0))
    np.random.seed(42)  # Deterministic for reproducibility
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        candidates.append(x0 + np.random.randn(2) * scale * x_scale)
    
    # Also try some fixed directions
    for angle in [0, np.pi/2, np.pi/4, 3*np.pi/4]:
        candidates.append(x0 + x_scale * np.array([np.cos(angle), np.sin(angle)]))
    
    # Try each candidate
    for cand in candidates:
        if not time_ok():
            break
        
        if not check_valid(cand):
            continue
        
        x_result, norm_result, calls = broyden_from(cand)
        
        if norm_result < best_norm:
            best_norm = norm_result
            best_x = x_result.copy()
        
        if best_norm <= 1e-8:
            break
    
    # Final check
    if check_valid(best_x):
        return best_x
    return x0
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using a global-to-local strategy.
    Phase 1: Coarse global search using derivative-free minimization of ||F(x)||^2.
    Phase 2: Local refinement using damped Newton with adaptive step size.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x0 = np.array(x0, dtype=float)
    
    def compute_F(val):
        return np.asarray(F(val), dtype=float)
    
    def objective(val):
        Fv = compute_F(val)
        if not np.all(np.isfinite(Fv)):
            return np.inf
        return np.dot(Fv, Fv)

    best_x = x0.copy()
    best_val = objective(x0)

    # Phase 1: Global search using Nelder-Mead on ||F(x)||^2
    # This is derivative-free and robust to poor initial guesses.
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None

    if minimize is not None and time_ok():
        # Set initial simplex size based on x0 scale
        x_scale = max(1.0, np.linalg.norm(x0))
        simplex_size = 0.1 * x_scale
        
        # Try multiple starting points if the first one fails
        start_points = [x0]
        # Add a few perturbations if we have time
        if time_ok():
            for scale in [0.5, 1.0, 5.0]:
                start_points.append(x0 + np.random.randn(2) * scale * x_scale)

        for sp in start_points:
            if not time_ok():
                break
            try:
                res = minimize(
                    objective, 
                    sp, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-6, 
                        'fatol': 1e-12,
                        'maxiter': 200,
                        'maxfev': 300
                    }
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
                    if best_val <= (1e-8)**2:
                        break
            except Exception:
                pass

    # Phase 2: Local refinement with Newton
    x = best_x.copy()
    h_base = 1e-7
    max_newton_iters = 50

    for it in range(max_newton_iters):
        if not time_ok():
            break
        
        Fx = compute_F(x)
        if not np.all(np.isfinite(Fx)):
            break
            
        norm_Fx_sq = np.dot(Fx, Fx)
        if norm_Fx_sq < best_val:
            best_val = norm_Fx_sq
            best_x = x.copy()
            
        if np.linalg.norm(Fx) <= 1e-10:
            break

        # Finite difference Jacobian
        h = h_base * max(1.0, np.linalg.norm(x)) ** 0.5
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (compute_F(x + e) - Fx) / h

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        # Damped step
        step_size = 1.0
        x_new = x + step_size * dx
        converged_step = False
        for _ in range(20):
            Fx_new = compute_F(x_new)
            norm_Fx_new_sq = np.dot(Fx_new, Fx_new)
            if norm_Fx_new_sq < norm_Fx_sq:
                converged_step = True
                break
            step_size *= 0.5
            x_new = x + step_size * dx
            if step_size < 1e-15:
                break
        
        if not converged_step:
            break
            
        x = x_new

    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x
    return x0
# EVOLVE-BLOCK-END

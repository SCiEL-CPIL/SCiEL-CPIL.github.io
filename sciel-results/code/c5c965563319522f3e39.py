"""Root finder for 3 fixed two-dimensional systems using Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's Good Method.
    
    Strategy:
    1. Estimate initial Jacobian via forward differences.
    2. Iterate Broyden's method:
       - Solve J @ dx = -F(x)
       - Line search to ensure descent in ||F(x)||
       - Update J using Broyden's rank-1 update
    3. Fallback to scipy.optimize.fsolve if Broyden fails.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()
    n = 2

    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    # Initial Evaluation
    if not time_ok():
        return best_x

    Fx = compute_F(x)
    if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
        return best_x
    
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx < best_norm:
        best_norm = norm_Fx
        best_x = x.copy()

    if norm_Fx <= 1e-8:
        return best_x

    # Estimate Initial Jacobian using Forward Differences
    # This costs 3 calls total (1 for Fx, 2 for perturbations)
    J = np.zeros((2, 2))
    h = 1e-7 * max(1.0, np.linalg.norm(x))
    
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        x_pert = x + e
        Fx_pert = compute_F(x_pert)
        if Fx_pert.shape != (2,) or not np.all(np.isfinite(Fx_pert)):
            # If perturbation fails, try central difference or just use identity?
            # Let's try central difference for robustness
            x_pert_neg = x - e
            Fx_pert_neg = compute_F(x_pert_neg)
            if Fx_pert_neg.shape == (2,) and np.all(np.isfinite(Fx_pert_neg)):
                J[:, j] = (Fx_pert - Fx_pert_neg) / (2 * h)
            else:
                # Fallback to identity if all fails
                J = np.eye(n)
                break
        else:
            J[:, j] = (Fx_pert - Fx) / h

    # Broyden's Method Loop
    max_iters = 100
    for it in range(max_iters):
        if not time_ok():
            break

        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize J if singular
            try:
                J_reg = J + 1e-8 * np.eye(n)
                dx = np.linalg.solve(J_reg, -Fx)
            except np.linalg.LinAlgError:
                break

        # Line Search
        alpha = 1.0
        x_new = x
        Fx_new = Fx
        found_step = False
        
        for _ in range(30):
            x_trial = x + alpha * dx
            Fx_trial = compute_F(x_trial)
            
            if Fx_trial.shape != (2,) or not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue

            norm_trial = np.linalg.norm(Fx_trial)
            
            if norm_trial < norm_Fx:
                x_new = x_trial
                Fx_new = Fx_trial
                found_step = True
                break
            
            alpha *= 0.5
            if alpha < 1e-10:
                break

        if not found_step:
            break

        # Update Jacobian (Broyden's Good Method)
        s = x_new - x
        y = Fx_new - Fx
        denom = np.dot(s, s)
        
        if denom > 1e-14:
            J = J + np.outer(y - J @ s, s) / denom
        
        x = x_new
        Fx = Fx_new
        norm_Fx = np.linalg.norm(Fx)

        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()

        if norm_Fx <= 1e-8:
            return best_x

    # Fallback: SciPy fsolve if Broyden didn't converge
    if best_norm > 1e-8 and time_ok():
        try:
            # Use the best point found so far as the starting point for fsolve
            sol, info, ier, msg = so.fsolve(
                lambda x: compute_F(x), 
                best_x, 
                full_output=True, 
                maxfev=200, 
                xtol=1e-10
            )
            if ier == 1:
                norm_sol = np.linalg.norm(compute_F(sol))
                if norm_sol < best_norm:
                    best_norm = norm_sol
                    best_x = sol.copy()
                if best_norm <= 1e-8:
                    return best_x
        except Exception:
            pass

    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)

# EVOLVE-BLOCK-END

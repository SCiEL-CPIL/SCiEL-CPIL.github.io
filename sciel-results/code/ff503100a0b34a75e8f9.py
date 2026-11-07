"""Root finder for 3 fixed 2D systems using homotopy continuation with Broyden refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _broyden_refine(F, x, max_iter=50, tol=1e-10):
    """
    Refine a root using Broyden's method with line search.
    Returns (x, norm_Fx, converged).
    """
    x = np.array(x, dtype=float).copy()
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return x, np.inf, False
    
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= tol:
        return x, norm_Fx, True
    
    # Initial Jacobian using forward differences
    J = np.zeros((2, 2))
    h = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h
    
    for _ in range(max_iter):
        # Solve for step direction
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize
            J_reg = J + 1e-10 * np.eye(2)
            try:
                d = np.linalg.solve(J_reg, -Fx)
            except np.linalg.LinAlgError:
                break
        
        # Line search
        alpha = 1.0
        step_accepted = False
        for _ in range(15):
            x_new = x + alpha * d
            Fx_new = F(x_new)
            
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            
            norm_Fx_new = np.linalg.norm(Fx_new)
            
            if norm_Fx_new <= tol:
                return x_new, norm_Fx_new, True
            
            # Armijo condition
            if norm_Fx_new < (1.0 - 1e-4 * alpha) * norm_Fx:
                # Update Broyden
                d_dot_d = np.dot(d, d)
                if d_dot_d > 1e-12:
                    delta_F = Fx_new - Fx - J @ d
                    J = J + np.outer(delta_F, d) / d_dot_d
                
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                step_accepted = True
                break
            else:
                alpha *= 0.5
        
        if not step_accepted:
            break
    
    return x, norm_Fx, norm_Fx <= tol


def _homotopy_primary(F, x0, deadline):
    """
    Homotopy continuation method: track root from simple problem to target.
    H(t, x) = (1-t)*A*x + t*F(x) = 0 where A is a simple invertible matrix.
    Start at t=0 (easy linear problem), continue to t=1 (target).
    """
    x0 = np.array(x0, dtype=float).copy()
    
    # Choose a simple reference matrix A. Use identity for simplicity.
    # H(0, x) = x = 0 has solution x = 0.
    # H(1, x) = F(x) = 0 is our target.
    # H(t, x) = (1-t)*x + t*F(x) = 0
    
    x = np.zeros(2)  # Root of H(0, x) = 0
    t = 0.0
    
    # Number of homotopy steps
    n_steps = 20
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        if time.time() > deadline:
            break
        
        t = (i + 1) * dt
        
        # Define homotopy function
        def H(x_val, t_val=t):
            return (1.0 - t_val) * x_val + t_val * F(x_val)
        
        # Use Broyden to solve H(t, x) = 0 starting from current x
        x, norm_H, converged = _broyden_refine(H, x, max_iter=30, tol=1e-12)
        
        if not converged and norm_H > 1e-8:
            # Try smaller step
            dt_half = dt / 2
            for _ in range(3):
                t_mid = t - dt_half / 2
                def H_mid(x_val, t_val=t_mid):
                    return (1.0 - t_val) * x_val + t_val * F(x_val)
                x, norm_H, converged = _broyden_refine(H_mid, x, max_iter=20, tol=1e-12)
                if converged:
                    break
                dt_half /= 2
    
    # Final check at t=1
    Fx = F(x)
    if np.all(np.isfinite(Fx)) and np.linalg.norm(Fx) <= 1e-8:
        return x, np.linalg.norm(Fx), True
    
    return x, np.linalg.norm(Fx) if np.all(np.isfinite(Fx)) else np.inf, False


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses homotopy continuation as primary strategy, with Broyden refinement.
    """
    x0_arr = np.array(x0, dtype=float).copy()
    if x0_arr.shape != (2,):
        x0_arr = x0_arr.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Strategy 1: Try direct Broyden from x0 (fast if x0 is good)
    x, norm_F, converged = _broyden_refine(F, x0_arr, max_iter=30, tol=1e-10)
    if converged:
        return x
    
    best_x = x
    best_norm = norm_F
    
    # Strategy 2: Homotopy continuation
    if time.time() < deadline:
        x_h, norm_h, conv_h = _homotopy_primary(F, x0_arr, deadline)
        if conv_h:
            return x_h
        if norm_h < best_norm:
            best_x = x_h
            best_norm = norm_h
    
    # Strategy 3: Try Broyden from homotopy result or x0 with perturbation
    if time.time() < deadline - 5.0:
        # Perturb x0 and try again
        for scale in [0.1, 0.5, 1.0, 2.0]:
            if time.time() > deadline:
                break
            x_try = x0_arr + scale * np.array([0.1, -0.1])
            x_r, norm_r, conv_r = _broyden_refine(F, x_try, max_iter=30, tol=1e-10)
            if conv_r:
                return x_r
            if norm_r < best_norm:
                best_x = x_r
                best_norm = norm_r
    
    # Strategy 4: Grid search for a better starting point, then Broyden
    if time.time() < deadline - 10.0:
        # Coarse grid around x0
        for dx in [-2, -1, -0.5, 0.5, 1, 2]:
            for dy in [-2, -1, -0.5, 0.5, 1, 2]:
                if time.time() > deadline:
                    break
                x_try = x0_arr + np.array([dx, dy])
                Fx_try = F(x_try)
                if np.all(np.isfinite(Fx_try)):
                    norm_try = np.linalg.norm(Fx_try)
                    if norm_try < best_norm:
                        best_x = x_try
                        best_norm = norm_try
                    # Try Broyden from this point
                    x_r, norm_r, conv_r = _broyden_refine(F, x_try, max_iter=20, tol=1e-10)
                    if conv_r:
                        return x_r
                    if norm_r < best_norm:
                        best_x = x_r
                        best_norm = norm_r
    
    return best_x

# EVOLVE-BLOCK-END

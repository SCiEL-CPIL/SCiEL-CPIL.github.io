"""Root finder for 3 fixed 2D systems using Powell's secant method with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8 using Powell's secant method
    with multi-start restarts for robustness.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    def F_eval(v):
        """Safely evaluate F and return (value, norm) or (None, inf)."""
        try:
            fv = F(v)
            fv = np.asarray(fv, dtype=float)
            if fv.shape != (2,) or not np.all(np.isfinite(fv)):
                return None, np.inf
            return fv, np.linalg.norm(fv)
        except:
            return None, np.inf
    
    best_x = x.copy()
    best_norm = np.inf
    
    # Initial evaluation
    F_x, best_norm = F_eval(x)
    if F_x is None:
        return np.zeros(2)
    
    if best_norm <= tol:
        return best_x
    
    def powell_secant(x_start, max_iters=30):
        """
        Powell's secant method for 2D systems.
        Uses the secant equation to maintain Jacobian inverse approximation.
        Requires exactly 1 F-call per iteration after initial setup.
        Returns (x_result, F_result, norm_result).
        """
        x_curr = x_start.copy()
        F_curr, n_curr = F_eval(x_curr)
        if F_curr is None:
            return x_curr, None, np.inf
        
        if n_curr <= tol:
            return x_curr, F_curr, n_curr
        
        # Initial step for secant: use small perturbation
        h = 1e-4
        x_prev = x_curr.copy()
        x_curr = x_curr + h * np.array([1.0, 0.0])
        
        F_prev = F_curr.copy()
        n_prev = n_curr
        
        F_curr, n_curr = F_eval(x_curr)
        if F_curr is None:
            return x_prev, F_prev, n_prev
        
        if n_curr <= tol:
            return x_curr, F_curr, n_curr
        
        # Build initial inverse Jacobian approximation using secant
        # For 2D, we need two secant directions
        # First secant: along e1
        dx1 = x_curr - x_prev
        dF1 = F_curr - F_prev
        
        # Second secant: along e2 from x_curr
        x_temp = x_curr + h * np.array([0.0, 1.0])
        F_temp, n_temp = F_eval(x_temp)
        if F_temp is None:
            return x_curr, F_curr, n_curr
        
        if n_temp <= tol:
            return x_temp, F_temp, n_temp
        
        dx2 = x_temp - x_curr
        dF2 = F_temp - F_curr
        
        # Build J from secant equations: J * dx_i = dF_i
        # So J = dF_matrix * dx_matrix^{-1}
        dx_mat = np.column_stack([dx1, dx2])
        dF_mat = np.column_stack([dF1, dF2])
        
        det_dx = dx_mat[0,0]*dx_mat[1,1] - dx_mat[0,1]*dx_mat[1,0]
        if abs(det_dx) < 1e-12:
            # Fallback: use identity-based initial guess
            J_inv = np.eye(2) * 0.01
        else:
            # J_inv = dx_mat * inv(dF_mat)
            det_dF = dF_mat[0,0]*dF_mat[1,1] - dF_mat[0,1]*dF_mat[1,0]
            if abs(det_dF) < 1e-12:
                J_inv = np.eye(2) * 0.01
            else:
                inv_dF = np.array([[dF_mat[1,1], -dF_mat[0,1]],
                                    [-dF_mat[1,0], dF_mat[0,0]]]) / det_dF
                J_inv = dx_mat @ inv_dF
        
        # Now iterate using secant updates
        for it in range(max_iters):
            if time_left() < 2.0:
                break
            
            if n_curr <= tol:
                return x_curr, F_curr, n_curr
            
            # Newton-like step using approximate inverse Jacobian
            s = J_inv @ (-F_curr)
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-14:
                break
            if s_norm > 1.0:
                s = s / s_norm
            
            # Line search with backtracking
            alpha = 1.0
            f_prev_sq = n_curr ** 2
            accepted = False
            
            for _ls in range(10):
                x_try = x_curr + alpha * s
                F_try, n_try = F_eval(x_try)
                if F_try is None:
                    alpha *= 0.5
                    continue
                
                if n_try <= tol:
                    return x_try, F_try, n_try
                
                if n_try ** 2 < f_prev_sq * 0.5 + 1e-14:
                    # Secant update for 2D: Sherman-Morrison-like
                    # J_new^{-1} = J_inv + (dx - J_inv*dF) * (something)
                    # For 2D, use the two most recent secant directions
                    
                    dx_new = x_try - x_curr
                    dF_new = F_try - F_curr
                    
                    # Update J_inv using secant equation
                    # We want J_new * dx_new = dF_new
                    # With J_inv approximation, the residual is:
                    # r = dF_new - J_new * dx_new ≈ dF_new - (J_inv^{-1} * J_inv) * dx_new
                    # Simpler: build new J_inv from last two secant pairs
                    
                    # Keep history of last two (dx, dF) pairs
                    # For 2D, we can rebuild J_inv from the two most recent secants
                    
                    # For now, use Broyden-like rank-1 update on J_inv
                    # delta = dF_new - J_inv^{-1} @ dx_new  (this is the residual)
                    # But we have J_inv, so J_approx = inv(J_inv)
                    
                    # Use the standard Broyden update on J_inv
                    # y = dF_new, s_vec = dx_new
                    # J_inv_new = J_inv + (s_vec - J_inv @ y) * y^T / (y^T @ y)
                    
                    J_inv_y = J_inv @ dF_new
                    y_dot_y = np.dot(dF_new, dF_new)
                    
                    if y_dot_y > 1e-16:
                        J_inv = J_inv + np.outer(dx_new - J_inv_y, dF_new) / y_dot_y
                    
                    x_curr = x_try
                    F_curr = F_try
                    n_curr = n_try
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
        
        return x_curr, F_curr, n_curr
    
    # Multi-start strategy
    num_restarts = 20
    used_starts = [x.copy()]
    
    for restart in range(num_restarts):
        if time_left() < 3.0:
            break
        
        if restart == 0:
            x_start = x.copy()
        elif restart < 8:
            # Deterministic directions around best_x
            angle = 2 * np.pi * restart / 8
            radius = 0.5 * (1 + restart * 0.3)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        elif restart < 14:
            # Random points in a wider region
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(1.0, 5.0)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Broad random search
            x_start = np.random.uniform(-10, 10, size=2)
        
        if not np.all(np.isfinite(x_start)):
            continue
        
        # Check if this start is close to a previous one
        is_dup = False
        for prev in used_starts:
            if np.linalg.norm(x_start - prev) < 1e-6:
                is_dup = True
                break
        if is_dup:
            continue
        used_starts.append(x_start)
        
        # Evaluate at start
        F_start, n_start = F_eval(x_start)
        if F_start is None:
            continue
        
        if n_start < best_norm:
            best_norm = n_start
            best_x = x_start.copy()
        
        if n_start <= tol:
            return best_x
        
        # Run Powell's secant method
        x_res, F_res, n_res = powell_secant(x_start, max_iters=25)
        
        if F_res is not None and n_res < best_norm:
            best_norm = n_res
            best_x = x_res.copy()
        
        if n_res <= tol:
            return best_x
    
    # Final polish: try a few more Newton steps near best_x if it's close
    if best_norm < 1e-4 and time_left() > 5.0:
        for polish in range(5):
            if time_left() < 3.0:
                break
            x_try = best_x + 0.1 * np.random.randn(2)
            F_try, n_try = F_eval(x_try)
            if F_try is None:
                continue
            if n_try < best_norm:
                best_norm = n_try
                best_x = x_try.copy()
            if n_try <= tol:
                return best_x
    
    return best_x
# EVOLVE-BLOCK-END

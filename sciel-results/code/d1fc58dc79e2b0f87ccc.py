# EVOLVE-BLOCK-START
import numpy as np


def _homotopy_solver(F, x0, max_iter=100, tol=1e-12, max_t_steps=50, min_dt=1e-4):
    """
    Homotopy/continuation method using Broyden's method with Jacobian reuse.
    Solves F_t(x) = t*F(x) + (1-t)*(x - x0) = 0 for t from 0 to 1.
    """
    x0 = np.array(x0, dtype=float)
    x = x0.copy()
    
    # At t=0, F_t(x) = x - x0. Root is x0.
    # Jacobian of F_t w.r.t x is t*J_F + (1-t)*I.
    # At t=0, J_t = I.
    J = np.eye(2)
    
    t = 0.0
    dt = 0.1  # Initial step size for t
    max_dt = 0.5
    
    # Pre-check: is x0 already a root of F?
    # We don't want to spend calls if we are already there, but we need to be careful.
    # The homotopy starts at t=0 where x=x0 is a root.
    # We will step t.
    
    for _ in range(max_t_steps):
        # Determine next t
        t_new = min(t + dt, 1.0)
        
        # Define the homotopy function for this t
        def F_t(x_val):
            Fx = np.asarray(F(x_val), dtype=float)
            return t_new * Fx + (1.0 - t_new) * (x_val - x0)
        
        # Solve F_t(x) = 0 starting from current x
        # Use Broyden's method with current J as initial guess
        # We need to be careful: J is the Jacobian of F_{t_old}.
        # Ideally we should update J to reflect the change in t, but Broyden's method
        # will adapt it via rank-1 updates as we iterate on x for the new t.
        
        x_new = x.copy()
        Fx_new = F_t(x_new)
        norm_Fx = np.linalg.norm(Fx_new)
        
        converged = False
        for _iter in range(20):  # Max Newton iterations per t step
            if norm_Fx < tol:
                converged = True
                break
            
            # Solve J * dx = -Fx
            try:
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                if abs(det) < 1e-14:
                    # Singular, try pinv or reset J?
                    # If J is bad, maybe reset to FD estimate?
                    # For now, try pinv.
                    dx = np.linalg.pinv(J) @ (-Fx_new)
                else:
                    dx = np.linalg.solve(J, -Fx_new)
            except np.linalg.LinAlgError:
                dx = np.linalg.pinv(J) @ (-Fx_new)
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Line search
            step = 1.0
            accepted = False
            for _ls in range(10):
                x_try = x_new + step * dx
                Fx_try = F_t(x_try)
                if np.all(np.isfinite(Fx_try)):
                    norm_Fx_try = np.linalg.norm(Fx_try)
                    # Armijo condition
                    if norm_Fx_try <= norm_Fx - 1e-4 * step * np.linalg.norm(dx) * norm_Fx:
                        # Update Broyden
                        s = x_try - x_new
                        y = Fx_try - Fx_new
                        # Broyden rank-1 update
                        # J_new = J + (y - J*s) s^T / (s^T s)
                        denom = np.dot(s, s)
                        if denom > 1e-14:
                            J = J + np.outer(y - J @ s, s) / denom
                        
                        x_new = x_try
                        Fx_new = Fx_try
                        norm_Fx = norm_Fx_try
                        accepted = True
                        break
                step *= 0.5
            
            if not accepted:
                break
            
            if norm_Fx < tol:
                converged = True
                break
        
        if converged:
            # Success, update t and x
            x = x_new
            t = t_new
            # Increase dt if successful
            if dt < max_dt:
                dt = min(dt * 1.5, max_dt)
            
            if t >= 1.0 - 1e-10:
                break
        else:
            # Failure, reduce dt
            dt *= 0.5
            if dt < min_dt:
                # Stuck, try direct Broyden from current x on F
                # Or just break and return best guess
                break
    
    # Final check
    F_final = np.asarray(F(x), dtype=float)
    if np.all(np.isfinite(F_final)) and np.linalg.norm(F_final) <= 1e-8:
        return x
    
    # If homotopy failed to reach t=1 or didn't converge, try direct Broyden from x
    # But we must be careful not to exceed call limits.
    # Let's do a quick direct Broyden from x.
    
    def _broyden_direct(F, x_init, max_iter=30, tol=1e-13, eps=1e-6):
        x = np.array(x_init, dtype=float)
        Fx = np.asarray(F(x), dtype=float)
        if not np.all(np.isfinite(Fx)):
            return x
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= tol:
            return x
        
        # Initialize J with FD
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fx_e = np.asarray(F(x + e), dtype=float)
            if not np.all(np.isfinite(Fx_e)):
                eps2 = eps * 0.01
                e[j] = eps2
                Fx_e = np.asarray(F(x + e), dtype=float)
                if not np.all(np.isfinite(Fx_e)):
                    return x
                J[:, j] = (Fx_e - Fx) / eps2
            else:
                J[:, j] = (Fx_e - Fx) / eps
        
        for _ in range(max_iter):
            try:
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                if abs(det) < 1e-14:
                    dx = np.linalg.pinv(J) @ (-Fx)
                else:
                    dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.pinv(J) @ (-Fx)
                except Exception:
                    return x
            if not np.all(np.isfinite(dx)):
                return x
            
            step = 1.0
            accepted = False
            for _ in range(15):
                x_new = x + step * dx
                Fx_new = np.asarray(F(x_new), dtype=float)
                if np.all(np.isfinite(Fx_new)):
                    norm_Fx_new = np.linalg.norm(Fx_new)
                    if norm_Fx_new <= norm_Fx - 1e-4 * step * np.linalg.norm(dx) * norm_Fx:
                        s = x_new - x
                        y = Fx_new - Fx
                        denom = np.dot(s, s)
                        if denom > 1e-14:
                            J = J + np.outer(y - J @ s, s) / denom
                        x = x_new
                        Fx = Fx_new
                        norm_Fx = norm_Fx_new
                        accepted = True
                        break
                step *= 0.5
            if not accepted:
                return x
            if norm_Fx <= tol:
                return x
        return x

    x_direct = _broyden_direct(F, x)
    F_direct = np.asarray(F(x_direct), dtype=float)
    if np.all(np.isfinite(F_direct)) and np.linalg.norm(F_direct) <= 1e-8:
        return x_direct
    
    # Return best candidate
    norm_hom = np.linalg.norm(F_final) if np.all(np.isfinite(F_final)) else np.inf
    norm_dir = np.linalg.norm(F_direct) if np.all(np.isfinite(F_direct)) else np.inf
    
    if norm_dir < norm_hom:
        return x_direct
    return x


def find_root_system(F, x0):
    """
    Wrapper for the homotopy solver.
    """
    x0 = np.array(x0, dtype=float)
    # Ensure x0 is finite
    if not np.all(np.isfinite(x0)):
        x0 = np.zeros(2)
        
    # Try homotopy
    x_result = _homotopy_solver(F, x0)
    
    # Final validation
    F_res = np.asarray(F(x_result), dtype=float)
    if np.all(np.isfinite(F_res)) and np.linalg.norm(F_res) <= 1e-8:
        return x_result
    
    # If it failed, maybe try a simpler direct Broyden from x0?
    # The homotopy solver already tries a direct Broyden fallback internally.
    # But just in case, we can return the result from the homotopy solver's best effort.
    return x_result
# EVOLVE-BLOCK-END

import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Uses Broyden's method with adaptive Jacobian restart:
    1. Initial Jacobian via finite differences.
    2. Rank-1 updates to Jacobian.
    3. If step fails to reduce residual, restart Jacobian.
    4. Strict call and time limits.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    call_count = 0
    max_calls = 480  # Leave buffer for safety

    def safe_F(x_in):
        nonlocal call_count
        if time.time() > deadline:
            raise TimeoutError()
        if call_count >= max_calls:
            raise TimeoutError()
        call_count += 1
        result = F(x_in)
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            result = result.flatten()[:2]
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result

    try:
        # 1. Initial Evaluation
        Fx = safe_F(x)
        fx_norm = np.linalg.norm(Fx)
        
        if fx_norm <= 1e-8:
            return x

        # 2. Initial Jacobian via Finite Differences
        def compute_J(x_curr, Fx_curr):
            nonlocal call_count
            if call_count + 2 > max_calls:
                raise TimeoutError()
            h = 1e-7
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = safe_F(x_curr + e)
                J[:, j] = (Fx_e - Fx_curr) / h
            return J

        J = compute_J(x, Fx)

        # Ensure J is invertible
        try:
            np.linalg.inv(J)
        except np.linalg.LinAlgError:
            J += np.eye(2) * 1e-6

        best_x = x.copy()
        best_norm = fx_norm
        best_Fx = Fx.copy()

        for _ in range(200):  # Max iterations
            if time.time() > deadline:
                break
            if call_count >= max_calls - 5:
                break

            if best_norm <= 1e-8:
                return best_x

            # Solve for step
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            # Limit step size to prevent wild jumps
            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx / step_norm
            elif step_norm < 1e-10:
                # Step is tiny, likely converged or singular J
                if fx_norm < best_norm:
                    best_norm = fx_norm
                    best_x = x.copy()
                    best_Fx = Fx.copy()
                # If step is tiny but not converged, J might be bad. Restart.
                J = compute_J(x, Fx)
                continue

            x_new = x + dx
            Fx_new = safe_F(x_new)
            fx_new_norm = np.linalg.norm(Fx_new)
            
            # Check for success
            if fx_new_norm <= 1e-8:
                return x_new

            # Check for improvement
            if fx_new_norm < fx_norm:
                # Good step, update Broyden
                s = x_new - x
                y = Fx_new - Fx
                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm
                
                s_sq = s @ s
                if s_sq > 1e-14:
                    J = J + np.outer(y - J @ s, s) / s_sq
                
                if fx_norm < best_norm:
                    best_norm = fx_norm
                    best_x = x.copy()
                    best_Fx = Fx.copy()
            else:
                # Step did not improve. Restart Jacobian at current x.
                # We keep x and Fx, just recompute J.
                J = compute_J(x, Fx)
                
                # If we are stuck and not improving, try a small perturbation?
                # For 2D, restarting J usually fixes it.
                # If we restart too many times without progress, we might fail.
                # But Broyden is robust.

    except TimeoutError:
        pass

    # Return the best point found
    return best_x

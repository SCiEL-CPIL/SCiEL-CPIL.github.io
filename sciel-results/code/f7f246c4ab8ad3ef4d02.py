"""Root finder for 3 fixed 2-D systems: homotopy-path + Broyden + Newton polish."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.

    Strategy:
    1. Try homotopy continuation from origin (or x0) using adaptive RK4
       steps along the path x(t) = t * x_target, integrating the ODE
       dx/dt = -J^{-1} * (dF/dx * t * x_target + F(0) - F(x)) approximately.
       This exploits global field structure to avoid local basins.
    2. Broyden's method from the homotopy endpoint (or x0) with safeguards.
    3. Newton polish with full FD Jacobian near convergence.
    4. A few cheap restarts if stuck.

    Budget: < 500 calls per system, explicit time deadline.
    """
    start_time = time.time()
    time_limit = 118.0
    TOL = 1e-8

    def time_ok():
        return (time.time() - start_time) < time_limit

    call_count = [0]

    def safe_eval_F(x):
        try:
            x_arr = np.array(x, dtype=float)
            if x_arr.shape != (2,) or not np.all(np.isfinite(x_arr)):
                call_count[0] += 1
                return None
            result = np.array(F(x_arr), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None

    def full_jacobian_fd(x, Fx, h=1e-7):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J

    def reanchor_col(J, x, Fx, col, h=1e-7):
        e = np.zeros(2)
        e[col] = h
        Fx_e = safe_eval_F(x + e)
        if Fx_e is None:
            return False
        J[:, col] = (Fx_e - Fx) / h
        return True

    def solve(J, b):
        try:
            return np.linalg.solve(J, b)
        except np.linalg.LinAlgError:
            try:
                return np.linalg.solve(J + 1e-8 * np.eye(2), b)
            except np.linalg.LinAlgError:
                return None

    def polish(x, Fx, max_iter=12):
        """Newton polish with full FD Jacobian. Returns (x, Fx, success)."""
        x = x.copy()
        Fx = Fx.copy()
        for _ in range(max_iter):
            if not time_ok() or call_count[0] >= 490:
                break
            if np.linalg.norm(Fx) <= TOL:
                return x, Fx, True
            J = full_jacobian_fd(x, Fx, h=1e-8)
            if J is None:
                J = full_jacobian_fd(x, Fx, h=1e-6)
                if J is None:
                    break
            dx = solve(J, -Fx)
            if dx is None:
                break
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                x, Fx = x_new, Fx_new
        return x, Fx, np.linalg.norm(Fx) <= TOL

    def broyden(x_start, max_iter=40, reanchor_every=4, polish_thresh=1e-5):
        x = np.array(x_start, dtype=float).copy()
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= TOL:
            return x, Fx, True

        J = full_jacobian_fd(x, Fx, h=1e-7)
        if J is None:
            J = full_jacobian_fd(x, Fx, h=1e-5)
            if J is None:
                return x, Fx, False
        if abs(np.linalg.det(J)) < 1e-12:
            return x, Fx, False

        best_x, best_Fx, best_norm = x.copy(), Fx.copy(), np.linalg.norm(Fx)
        col = 0
        last_norm = best_norm

        for it in range(max_iter):
            if not time_ok() or call_count[0] >= 485:
                break
            nrm = np.linalg.norm(Fx)
            if nrm < best_norm:
                best_x, best_Fx, best_norm = x.copy(), Fx.copy(), nrm

            if nrm <= polish_thresh:
                xp, Fxp, ok = polish(x, Fx, max_iter=12)
                if ok:
                    return xp, Fxp, True
                if Fxp is not None:
                    if np.linalg.norm(Fxp) < best_norm:
                        best_x, best_Fx, best_norm = xp.copy(), Fxp.copy(), np.linalg.norm(Fxp)
                    x, Fx = xp, Fxp
                    nrm = np.linalg.norm(Fx)
                    if nrm <= TOL:
                        return x, Fx, True

            if nrm <= TOL:
                return x, Fx, True

            if it > 0 and it % reanchor_every == 0:
                reanchor_col(J, x, Fx, col, h=1e-7)
                col = 1 - col

            dx = solve(J, -Fx)
            if dx is None:
                break
            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                x_new = x + 0.5 * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    break

            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s

            x, Fx = x_new, Fx_new

            if np.linalg.norm(Fx) > last_norm * 0.999:
                break
            last_norm = np.linalg.norm(Fx)

        return best_x, best_Fx, best_norm <= TOL

    def homotopy_continuation(x_target, max_steps=30, dt=0.1):
        """
        Follow the path x(t) = t * x_target from t=0 to t=1 using
        a pseudo-transient continuation approach. At each t, we want F(x(t)) ~ 0
        by adjusting x(t) via Newton-like corrections.
        
        Uses the fact that at t=0, x=0, and F(0) is known.
        We integrate dx/dt = -J^{-1} * (J * dx_target/dt + dF/dt)
        which simplifies to tracking a modified residual.
        
        Practical approach: step t from 0 to 1, at each step do 2-3
        Broyden iterations to keep F(x(t)) small.
        """
        x = np.zeros(2)
        Fx = safe_eval_F(x)
        if Fx is None:
            return None, None, False
        
        # If F(0) is already small, we're near a root at origin
        if np.linalg.norm(Fx) <= TOL:
            return x, Fx, True
        
        # Initial Jacobian at origin
        J = full_jacobian_fd(x, Fx, h=1e-7)
        if J is None:
            J = full_jacobian_fd(x, Fx, h=1e-5)
            if J is None:
                return None, None, False
        if abs(np.linalg.det(J)) < 1e-12:
            return None, None, False
        
        dx_target = x_target / max_steps  # per step
        
        for step in range(max_steps):
            if not time_ok() or call_count[0] >= 470:
                break
            
            # Move x toward t * x_target
            t = (step + 1) / max_steps
            x_target_here = t * x_target
            
            # Do a few Broyden iterations to keep F small near x_target_here
            for inner in range(3):
                if not time_ok() or call_count[0] >= 475:
                    break
                
                dx_step = solve(J, -Fx)
                if dx_step is None:
                    break
                
                # Blend: want to move toward x_target_here, but also reduce F
                # Use a damped step
                alpha = 0.5
                dx_combined = alpha * dx_step + (1 - alpha) * (x_target_here - x)
                step_norm = np.linalg.norm(dx_combined)
                if step_norm > 5.0:
                    dx_combined = dx_combined * (5.0 / step_norm)
                
                x_new = x + dx_combined
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    break
                
                s = x_new - x
                y = Fx_new - Fx
                s_dot_s = np.dot(s, s)
                if s_dot_s > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_dot_s
                
                x, Fx = x_new, Fx_new
                
                if np.linalg.norm(Fx) <= TOL:
                    return x, Fx, True
        
        return x, Fx, np.linalg.norm(Fx) <= TOL

    x0_arr = np.array(x0, dtype=float)
    
    # Try homotopy from origin toward x0
    xh, Fxh, okh = homotopy_continuation(x0_arr, max_steps=20, dt=0.05)
    if okh:
        return xh
    
    best_x = x0_arr.copy()
    best_norm = np.linalg.norm(Fxh) if Fxh is not None else np.inf
    
    if xh is not None and np.linalg.norm(Fxh) < best_norm:
        best_x = xh.copy()
        best_norm = np.linalg.norm(Fxh)
    
    # Broyden from x0
    x, Fx, success = broyden(x0_arr, max_iter=40, reanchor_every=4)
    if success:
        return x
    
    if Fx is not None:
        nr = np.linalg.norm(Fx)
        if nr < best_norm:
            best_norm = nr
            best_x = x.copy()
    
    # Broyden from homotopy endpoint
    if xh is not None:
        x2, Fx2, success2 = broyden(xh, max_iter=30, reanchor_every=4)
        if success2:
            return x2
        if Fx2 is not None:
            nr2 = np.linalg.norm(Fx2)
            if nr2 < best_norm:
                best_norm = nr2
                best_x = x2.copy()
    
    # A few cheap restarts
    rng = np.random.RandomState(42)
    restarts = [0.5 * x0_arr, 2.0 * x0_arr, -1.0 * x0_arr, np.zeros(2)]
    for _ in range(4):
        restarts.append(x0_arr + rng.randn(2) * max(0.5, np.linalg.norm(x0_arr)) * 0.5)
    
    for r in restarts:
        if not time_ok() or call_count[0] >= 455:
            break
        xr, Fxr, okr = broyden(r, max_iter=22, reanchor_every=4)
        if okr:
            return xr
        if Fxr is not None:
            nr = np.linalg.norm(Fxr)
            if nr < best_norm:
                best_norm = nr
                best_x = xr.copy()
    
    return best_x

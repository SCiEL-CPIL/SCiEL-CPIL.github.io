"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using a robust hybrid approach:
    1. Broyden's method (good variant) with finite-difference initial Jacobian.
    2. Backtracking line search for step acceptance.
    3. Final polish phase with finite-difference Newton steps for high precision.
    4. Time-guarded: return best result before deadline.
    
    Budget: <500 calls to F.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _time_ok():
        return time.time() < deadline

    def _call_F(x_val):
        nonlocal call_count
        if not _time_ok():
            return None
        if call_count >= 495:  # leave 5 calls for safety
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _fd_jacobian(x_val, Fx_val, h=1e-5):
        """Compute Jacobian via forward differences. Uses 2 calls (Fx already known)."""
        J = np.zeros((2, 2))
        for j in range(2):
            if not _time_ok() or call_count >= 495:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x_val + e)
            if Fx_plus is not None:
                J[:, j] = (Fx_plus - Fx_val) / h
            else:
                J[:, j] = 1.0  # fallback
        return J

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian
    J = _fd_jacobian(x, Fx)
    
    # Ensure J is well-conditioned
    try:
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-10:
            J += np.eye(2) * 1e-6
    except np.linalg.LinAlgError:
        J = np.eye(2)

    # Broyden's method main loop
    max_iter = 400
    stalled_count = 0
    prev_norm = norm_Fx

    for it in range(max_iter):
        if not _time_ok() or call_count >= 490:
            break

        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize if singular
            try:
                J_reg = J + 1e-8 * np.eye(2)
                d = np.linalg.solve(J_reg, -Fx)
            except np.linalg.LinAlgError:
                break

        if not np.all(np.isfinite(d)):
            break

        # Limit step size
        norm_d = np.linalg.norm(d)
        max_step = 10.0
        if norm_d > max_step:
            d = d / norm_d * max_step

        # Backtracking line search
        alpha = 1.0
        norm_current = norm_Fx
        found_step = False
        x_old = x.copy()
        Fx_old = Fx.copy()

        for _ls in range(15):
            if not _time_ok() or call_count >= 495:
                break

            x_new = x + alpha * d
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue

            Fx_new = _call_F(x_new)
            if Fx_new is None:
                alpha *= 0.5
                continue

            norm_new = np.linalg.norm(Fx_new)

            if norm_new < norm_current:
                # Accept step
                x = x_new
                Fx = Fx_new
                found_step = True
                break
            else:
                alpha *= 0.5

        if not found_step:
            # Try a small gradient step
            try:
                grad = J.T @ Fx
                norm_grad = np.linalg.norm(grad)
                if norm_grad > 1e-15:
                    step = -grad / norm_grad * 0.01
                    x_new = x + step
                    if np.all(np.all(np.isfinite(x_new))):
                        Fx_new = _call_F(x_new)
                        if Fx_new is not None:
                            norm_new = np.linalg.norm(Fx_new)
                            if norm_new < norm_Fx:
                                x = x_new
                                Fx = Fx_new
                                found_step = True
            except Exception:
                pass

            if not found_step:
                # Stalled
                stalled_count += 1
                if stalled_count >= 3:
                    break
                continue

        # Update best
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x.copy()

        if best_norm <= 1e-8:
            break

        # Broyden rank-1 update
        s = x - x_old
        y = Fx - Fx_old
        s_dot_s = np.dot(s, s)
        if s_dot_s > 1e-16:
            J = J + np.outer(y - J @ s, s) / s_dot_s

        # Check for stagnation
        if norm_Fx - norm_new < 1e-12:
            stalled_count += 1
        else:
            stalled_count = 0

        prev_norm = norm_Fx

    # Final polish: if close, do a few more Newton steps
    if best_norm < 1e-4:
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            for _ in range(8):
                if not _time_ok() or call_count >= 495:
                    break
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx <= 1e-8:
                    return x
                # Recompute Jacobian for polish
                h = 1e-7
                J_polish = _fd_jacobian(x, Fx, h)
                try:
                    dx = np.linalg.solve(J_polish, -Fx)
                    x_new = x + dx
                    if np.all(np.isfinite(x_new)):
                        Fx_new = _call_F(x_new)
                        if Fx_new is not None:
                            if np.linalg.norm(Fx_new) < norm_Fx:
                                x = x_new
                                Fx = Fx_new
                                if np.linalg.norm(Fx) < best_norm:
                                    best_x = x.copy()
                                    best_norm = np.linalg.norm(Fx)
                except np.linalg.LinAlgError:
                    break

    return best_x
# EVOLVE-BLOCK-END

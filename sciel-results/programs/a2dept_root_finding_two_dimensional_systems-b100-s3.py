"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8.
    
    Strategy:
      1. Broyden's method with forward-difference initial Jacobian.
         - Only 1 function call per iteration after the initial 3 calls.
         - This is significantly more call-efficient than Newton+FD (3 calls/iter).
      2. If Broyden stalls or fails, fall back to damped Newton.
      3. Time-budgeted with explicit deadline guard.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 105.0  # Safety margin before 120s limit
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        if call_count >= 495:
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

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian via forward differences (2 more calls)
    h = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        if Fx_plus is None:
            Fx_plus = Fx  # Fallback
        J[:, j] = (Fx_plus - Fx) / h

    # Ensure J is reasonably invertible
    det_J = np.linalg.det(J)
    if abs(det_J) < 1e-10:
        J += np.eye(2) * 1e-8

    # Broyden's method (good-far variant)
    for it in range(480):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break

        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                break

        # Step control: limit step size to avoid wild excursions
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 5.0:
            dx = dx / norm_dx * 5.0

        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            dx = dx * 0.1
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break

        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break

        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        if norm_Fx_new <= 1e-8:
            return x_new

        # Broyden rank-1 update (good-far)
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        if s_norm_sq > 1e-16:
            J = J + np.outer(y - J @ s, s) / s_norm_sq

        x = x_new
        Fx = Fx_new

        # Stagnation detection: if no progress for 10 iterations, break
        if it > 10 and norm_Fx_new >= best_norm * 0.9999:
            # Try a damped step
            pass

    # If Broyden didn't converge, try damped Newton from best point
    if best_norm > 1e-8:
        x = best_x.copy()
        for it in range(40):
            if time.time() > deadline:
                break
            if call_count >= 490:
                break

            Fx = _call_F(x)
            if Fx is None:
                break
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-8:
                return x

            # Compute Jacobian via forward differences
            h = 1e-6
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_plus = _call_F(x + e)
                if Fx_plus is None:
                    break
                J[:, j] = (Fx_plus - Fx) / h
            else:
                # Backtracking line search
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break

                step = 1.0
                for _ in range(20):
                    x_new = x + step * dx
                    Fx_new = _call_F(x_new)
                    if Fx_new is None:
                        break
                    norm_new = np.linalg.norm(Fx_new)
                    if norm_new < norm_Fx:
                        x = x_new
                        if norm_new < best_norm:
                            best_x = x.copy()
                            best_norm = norm_new
                        if norm_new <= 1e-8:
                            return x
                        break
                    step *= 0.5
                else:
                    break

    return best_x
# EVOLVE-BLOCK-END

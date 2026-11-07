# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's quasi-Newton method
    with adaptive damping, a central-difference polish phase, and a time guard.
    """
    start_time = time.time()
    TIME_LIMIT = 145.0  # Safety margin before 150s kill

    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    def safe_eval_F(v):
        """Evaluate F with safety checks, returning (Fx, success)."""
        try:
            Fx = F(v)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None, False
            return Fx, True
        except Exception:
            return None, False

    x = np.array(x0, dtype=float).copy()

    # Check initial point
    Fx, ok = safe_eval_F(x)
    if not ok:
        x = _local_search(F, x, time_ok)
        Fx, ok = safe_eval_F(x)
        if not ok:
            return x
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Initialize Jacobian estimate using finite differences
    J = np.zeros((2, 2))
    h = 1e-6
    try:
        x0_pert = x.copy()
        x0_pert[0] += h
        Fx0, ok0 = safe_eval_F(x0_pert)
        if ok0:
            J[:, 0] = (Fx0 - Fx) / h
        else:
            J[:, 0] = np.array([1.0, 0.0])

        x1_pert = x.copy()
        x1_pert[1] += h
        Fx1, ok1 = safe_eval_F(x1_pert)
        if ok1:
            J[:, 1] = (Fx1 - Fx) / h
        else:
            J[:, 1] = np.array([0.0, 1.0])
    except Exception:
        J = np.eye(2)

    # Broyden's method (Good variant) with accepting any residual-reducing step
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    for _ in range(450):
        if not time_ok():
            break

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        # Damping: accept any step that reduces residual, backtrack if needed
        alpha = 1.0
        Fx_norm = np.linalg.norm(Fx)
        x_new = x + alpha * dx
        Fx_new, ok = safe_eval_F(x_new)

        if not ok:
            # Try smaller steps
            for _ in range(8):
                alpha *= 0.5
                x_new = x + alpha * dx
                Fx_new, ok = safe_eval_F(x_new)
                if ok:
                    break
            if not ok:
                break

        Fx_new_norm = np.linalg.norm(Fx_new)

        # Check convergence
        if Fx_new_norm <= 1e-8:
            return x_new

        # Backtracking if needed (only if step didn't reduce residual)
        if Fx_new_norm >= Fx_norm:
            for _ in range(8):
                alpha *= 0.5
                x_new = x + alpha * dx
                Fx_new, ok = safe_eval_F(x_new)
                if not ok:
                    break
                Fx_new_norm = np.linalg.norm(Fx_new)
                if Fx_new_norm < Fx_norm:
                    break
            if not ok or Fx_new_norm >= Fx_norm:
                # Accept anyway if it's the best we can do
                if Fx_new_norm < best_norm:
                    best_x = x_new.copy()
                    best_norm = Fx_new_norm
                break

        # Update Broyden's Jacobian (Good variant)
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = s @ s
        if s_norm_sq > 1e-15:
            J = J + (np.outer(y, s) - np.outer(J @ s, s)) / s_norm_sq

        x = x_new
        Fx = Fx_new

        if Fx_new_norm < best_norm:
            best_x = x_new.copy()
            best_norm = Fx_new_norm

    # Central-difference polish phase for high-precision convergence
    if best_norm > 1e-8 and time_ok():
        x = best_x.copy()
        Fx, ok = safe_eval_F(x)
        if ok:
            Fx_norm = np.linalg.norm(Fx)
            if Fx_norm > 1e-8:
                # Use central differences for better Jacobian
                h_c = 1e-5
                J_c = np.zeros((2, 2))
                for i in range(2):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += h_c
                    xm[i] -= h_c
                    Fxp, okp = safe_eval_F(xp)
                    Fxm, okm = safe_eval_F(xm)
                    if okp and okm:
                        J_c[:, i] = (Fxp - Fxm) / (2 * h_c)
                    else:
                        # Fallback to forward difference
                        Fxp, okp = safe_eval_F(xp)
                        if okp:
                            J_c[:, i] = (Fxp - Fx) / h_c
                        else:
                            J_c[:, i] = np.eye(2)[:, i]

                # Newton iterations with central-difference Jacobian updates
                for _ in range(30):
                    if not time_ok():
                        break
                    try:
                        dx = np.linalg.solve(J_c, -Fx)
                    except np.linalg.LinAlgError:
                        try:
                            dx = np.linalg.lstsq(J_c, -Fx, rcond=None)[0]
                        except Exception:
                            break

                    x_new = x + dx
                    Fx_new, ok = safe_eval_F(x_new)
                    if not ok:
                        # Backtrack
                        alpha = 0.5
                        for _ in range(5):
                            x_new = x + alpha * dx
                            Fx_new, ok = safe_eval_F(x_new)
                            if ok:
                                break
                            alpha *= 0.5
                        if not ok:
                            break

                    Fx_new_norm = np.linalg.norm(Fx_new)
                    if Fx_new_norm <= 1e-8:
                        return x_new

                    # Update Jacobian with central differences
                    h_c = 1e-5
                    for i in range(2):
                        xp = x_new.copy()
                        xm = x_new.copy()
                        xp[i] += h_c
                        xm[i] -= h_c
                        Fxp, okp = safe_eval_F(xp)
                        Fxm, okm = safe_eval_F(xm)
                        if okp and okm:
                            J_c[:, i] = (Fxp - Fxm) / (2 * h_c)
                        elif okp:
                            J_c[:, i] = (Fxp - Fx_new) / h_c

                    x = x_new
                    Fx = Fx_new

                    if Fx_new_norm < best_norm:
                        best_x = x_new.copy()
                        best_norm = Fx_new_norm

    return best_x


def _local_search(F, x, time_ok):
    """Simple local search to find a point where F is finite."""
    x = np.array(x, dtype=float)
    for dx in [1e-2, 1e-1, 1.0, 10.0]:
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            if not time_ok():
                return x
            step = dx * np.array([np.cos(angle), np.sin(angle)])
            x_try = x + step
            try:
                Fx = F(x_try)
                Fx = np.asarray(Fx, dtype=float)
                if Fx.shape != (2,):
                    Fx = Fx.reshape(2)
                if np.all(np.isfinite(Fx)):
                    x = x_try
                    return x
            except Exception:
                continue
    return x
# EVOLVE-BLOCK-END

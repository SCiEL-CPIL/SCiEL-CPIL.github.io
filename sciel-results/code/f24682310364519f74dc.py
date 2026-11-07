"""Root finder for 3 fixed 2-D systems: Broyden's method with safeguards + Newton polish."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.

    Strategy:
    1. Broyden's method from x0 with cheap (1-column) periodic Jacobian re-anchoring.
    2. When close to a root, switch to a Newton polish (full FD Jacobian) for
       superlinear convergence to 1e-8.
    3. If stuck, a few cheap random restarts.

    Budget: < 500 calls per system, explicit time deadline with safety margin.
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
        """Recompute one column of J via FD (1 call)."""
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

        # Initial Jacobian (2 calls)
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

            # Polish when close: superlinear convergence to 1e-8
            if nrm <= polish_thresh:
                xp, Fxp, ok = polish(x, Fx, max_iter=12)
                if ok:
                    return xp, Fxp, True
                # adopt polish result as new state
                if Fxp is not None:
                    if np.linalg.norm(Fxp) < best_norm:
                        best_x, best_Fx, best_norm = xp.copy(), Fxp.copy(), np.linalg.norm(Fxp)
                    x, Fx = xp, Fxp
                    nrm = np.linalg.norm(Fx)
                    if nrm <= TOL:
                        return x, Fx, True

            if nrm <= TOL:
                return x, Fx, True

            # Cheap re-anchoring: 1 column per re-anchor
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

            # Broyden rank-1 update
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s

            x, Fx = x_new, Fx_new

            # Stagnation check
            if np.linalg.norm(Fx) > last_norm * 0.999:
                break
            last_norm = np.linalg.norm(Fx)

        return best_x, best_Fx, best_norm <= TOL

    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden(x0_arr, max_iter=40, reanchor_every=4)
    if success:
        return x

    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf

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

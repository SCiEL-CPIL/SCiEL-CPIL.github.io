"""Root finder using homotopy continuation with Jacobian reuse and Newton steps."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def time_left():
        return deadline - time.time()

    x = np.array(x0, dtype=float, copy=True)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    best_x = x.copy()
    best_norm = np.inf

    def safe_call(F_func, x_pt):
        try:
            result = F_func(x_pt)
            result = np.asarray(result, dtype=float)
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None

    # Initial evaluation
    Fx = safe_call(F, x)
    if Fx is None:
        return x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_norm = norm_Fx
    best_x = x.copy()

    # Compute initial Jacobian using central differences
    def compute_jacobian(x_pt, Fx_pt):
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_call(F, x_pt + e)
            Fx_minus = safe_call(F, x_pt - e)
            if Fx_plus is not None and Fx_minus is not None:
                J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
            else:
                # Fallback to forward difference
                Fx_plus = safe_call(F, x_pt + e)
                if Fx_plus is not None:
                    J[:, j] = (Fx_plus - Fx_pt) / h
                else:
                    # Try larger step
                    h2 = 1e-4
                    e2 = np.zeros(2)
                    e2[j] = h2
                    Fx_plus2 = safe_call(F, x_pt + e2)
                    if Fx_plus2 is not None:
                        J[:, j] = (Fx_plus2 - Fx_pt) / h2
                    else:
                        J[:, j] = 1.0
        return J

    # Newton step with backtracking line search
    def newton_step(x_pt, Fx_pt, J, max_backtracks=10):
        try:
            step = np.linalg.solve(J, -Fx_pt)
        except np.linalg.LinAlgError:
            try:
                step = np.linalg.lstsq(J, -Fx_pt, rcond=None)[0]
            except Exception:
                return None, None, None

        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            return None, None, None

        # Cap step size
        if step_norm > 1.0:
            step = step / step_norm * 1.0

        for backtrack in range(max_backtracks):
            alpha = 1.0 / (2 ** backtrack)
            x_new = x_pt + alpha * step
            Fx_new = safe_call(F, x_new)
            if Fx_new is None:
                continue
            norm_new = np.linalg.norm(Fx_new)
            # Accept if we made progress
            if norm_new < np.linalg.norm(Fx_pt) * 0.5:
                return x_new, Fx_new, norm_new
            # Accept if step is very small
            if alpha * step_norm < 1e-12:
                return x_new, Fx_new, norm_new
        return None, None, None

    # Broyden update
    def broyden_update(J, x_old, Fx_old, x_new, Fx_new):
        delta_x = x_new - x_old
        delta_F = Fx_new - Fx_old
        dot = np.dot(delta_x, delta_x)
        if dot > 1e-15:
            J = J + np.outer(delta_F - J @ delta_x, delta_x) / dot
        return J

    # Strategy 1: Pure Broyden's method
    def try_broyden(x_start, Fx_start, J_start, max_iter=80):
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        J_cur = J_start.copy()
        norm_cur = np.linalg.norm(Fx_cur)
        best_local_x = x_cur.copy()
        best_local_norm = norm_cur

        for _ in range(max_iter):
            if time_left() < 3.0:
                break
            if norm_cur <= 1e-8:
                return x_cur, norm_cur

            x_new, Fx_new, norm_new = newton_step(x_cur, Fx_cur, J_cur)
            if x_new is None:
                # Try smaller perturbation
                x_pert = x_cur + 0.1 * np.random.randn(2)
                Fx_pert = safe_call(F, x_pert)
                if Fx_pert is None:
                    break
                norm_pert = np.linalg.norm(Fx_pert)
                if norm_pert < best_local_norm:
                    best_local_x = x_pert.copy()
                    best_local_norm = norm_pert
                break

            # Update Jacobian with Broyden
            J_cur = broyden_update(J_cur, x_cur, Fx_cur, x_new, Fx_new)

            x_cur = x_new
            Fx_cur = Fx_new
            norm_cur = norm_new

            if norm_cur < best_local_norm:
                best_local_norm = norm_cur
                best_local_x = x_cur.copy()

        return best_local_x, best_local_norm

    # Strategy 2: Homotopy continuation - gradually modify the problem
    def try_homotopy(x_start, Fx_start, J_start):
        """
        Use homotopy: H(x, t) = (1-t)*F(x) + t*F_simple(x)
        where F_simple is a simple system we can solve easily.
        Start at t=0 (easy) and move to t=1 (target).
        """
        # Define a simple system: F_simple(x) = x (identity)
        # H(x, t) = (1-t)*F(x) + t*x
        # At t=0: H = F (target)
        # At t=1: H = x (root at origin)
        # We go from t=1 to t=0

        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        J_cur = J_start.copy()
        norm_cur = np.linalg.norm(Fx_cur)
        best_local_x = x_cur.copy()
        best_local_norm = norm_cur

        # Start from t=1 (easy) and go to t=0 (hard)
        # But we start at x0, so let's try different t values
        t_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.0, 0.0]

        for t in t_values:
            if time_left() < 3.0:
                break

            # Evaluate H(x, t) = (1-t)*F(x) + t*x
            def H(x_pt, t_val=t):
                Fx_pt = safe_call(F, x_pt)
                if Fx_pt is None:
                    return None
                return (1 - t_val) * Fx_pt + t_val * x_pt

            # Evaluate at current x
            Hx = H(x_cur)
            if Hx is None:
                break
            norm_H = np.linalg.norm(Hx)

            if norm_H <= 1e-10:
                # Check if we're close to the actual root
                Fx_check = safe_call(F, x_cur)
                if Fx_check is not None:
                    norm_check = np.linalg.norm(Fx_check)
                    if norm_check < best_local_norm:
                        best_local_norm = norm_check
                        best_local_x = x_cur.copy()
                continue

            # Compute Jacobian of H at current x
            # J_H = (1-t)*J_F + t*I
            if t < 1.0:
                J_H = (1 - t) * J_cur + t * np.eye(2)
            else:
                J_H = np.eye(2)

            # Newton step on H
            try:
                step = np.linalg.solve(J_H, -Hx)
            except np.linalg.LinAlgError:
                try:
                    step = np.linalg.lstsq(J_H, -Hx, rcond=None)[0]
                except Exception:
                    break

            step_norm = np.linalg.norm(step)
            if step_norm < 1e-15:
                break
            if step_norm > 0.5:
                step = step / step_norm * 0.5

            x_new = x_cur + step
            Hx_new = H(x_new)
            if Hx_new is None:
                break
            norm_H_new = np.linalg.norm(Hx_new)

            # Update Jacobian
            if t < 1.0:
                delta_x = x_new - x_cur
                delta_H = Hx_new - Hx
                dot = np.dot(delta_x, delta_x)
                if dot > 1e-15:
                    J_H = J_H + np.outer(delta_H - J_H @ delta_x, delta_x) / dot
                # Extract J_F from J_H: J_H = (1-t)*J_F + t*I => J_F = (J_H - t*I)/(1-t)
                if t < 1.0:
                    J_cur = (J_H - t * np.eye(2)) / (1 - t)

            x_cur = x_new

            # Check actual F norm
            Fx_check = safe_call(F, x_cur)
            if Fx_check is not None:
                norm_check = np.linalg.norm(Fx_check)
                if norm_check < best_local_norm:
                    best_local_norm = norm_check
                    best_local_x = x_cur.copy()

            if norm_check <= 1e-8:
                return x_cur, norm_check

        return best_local_x, best_local_norm

    # Compute initial Jacobian
    J = compute_jacobian(x, Fx)

    # Try Broyden first
    x_broyden, norm_broyden = try_broyden(x, Fx, J)
    if norm_broyden < best_norm:
        best_norm = norm_broyden
        best_x = x_broyden.copy()

    if best_norm <= 1e-8:
        return best_x

    # Try homotopy
    if time_left() > 5.0:
        x_homo, norm_homo = try_homotopy(x, Fx, J)
        if norm_homo < best_norm:
            best_norm = norm_homo
            best_x = x_homo.copy()

    if best_norm <= 1e-8:
        return best_x

    # Try multiple random restarts with Broyden
    if time_left() > 5.0:
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time_left() < 3.0:
                break
            x_restart = best_x + 0.1 * rng.randn(2)
            Fx_restart = safe_call(F, x_restart)
            if Fx_restart is None:
                continue
            norm_restart = np.linalg.norm(Fx_restart)
            if norm_restart >= best_norm:
                continue
            J_restart = compute_jacobian(x_restart, Fx_restart)
            x_new, norm_new = try_broyden(x_restart, Fx_restart, J_restart, max_iter=50)
            if norm_new < best_norm:
                best_norm = norm_new
                best_x = x_new.copy()
                if best_norm <= 1e-8:
                    break

    return best_x
# EVOLVE-BLOCK-END

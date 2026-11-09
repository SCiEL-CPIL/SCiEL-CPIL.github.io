"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Robust root finder for 2D systems.
    Strategy:
    1. Try damped Newton with forward-difference Jacobian (3 calls per iteration).
    2. If Newton stalls or fails, try Levenberg-Marquardt.
    3. If still failing, try a few random restarts with Newton.
    4. Return the best x found (lowest ||F(x)||) that is finite.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()
    n = 2

    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    # Helper: forward-difference Jacobian (3 calls: F(x), F(x+e1), F(x+e2))
    def finite_diff_jacobian_forward(x_current, h):
        Fx = compute_F(x_current)
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (compute_F(x_current + e) - Fx) / h
        return J, Fx

    # Helper: central-difference Jacobian (5 calls: F(x), F(x+e1), F(x-e1), F(x+e2), F(x-e2))
    def finite_diff_jacobian_central(x_current, h):
        Fx = compute_F(x_current)
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (compute_F(x_current + e) - compute_F(x_current - e)) / (2 * h)
        return J, Fx

    # Method 1: Damped Newton with adaptive step size
    max_newton_iters = 100
    h_base = 1e-7

    for it in range(max_newton_iters):
        if not time_ok():
            break
        Fx = compute_F(x)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()
        if norm_Fx <= 1e-10:
            break

        # Adaptive step size for finite differences
        h = h_base * max(1.0, np.linalg.norm(x)) ** 0.5
        J, Fx = finite_diff_jacobian_forward(x, h)

        # Check if Jacobian is well-conditioned
        try:
            cond = np.linalg.cond(J)
        except np.linalg.LinAlgError:
            cond = np.inf

        if cond > 1e10:
            # Ill-conditioned; try Levenberg-Marquardt
            try:
                JtJ = J.T @ J
                lambda_lq = 1e-6
                while True:
                    try:
                        dx = np.linalg.solve(JtJ + lambda_lq * np.eye(2), -J.T @ Fx)
                        break
                    except np.linalg.LinAlgError:
                        lambda_lq *= 10
                        if lambda_lq > 1e12:
                            raise
            except Exception:
                break
        else:
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

        # Damped step with backtracking line search
        step_size = 1.0
        x_new = x + step_size * dx
        for _ in range(30):
            Fx_new = compute_F(x_new)
            norm_Fx_new = np.linalg.norm(Fx_new)
            if norm_Fx_new < norm_Fx:
                break
            step_size *= 0.5
            x_new = x + step_size * dx
            if step_size < 1e-15:
                break
        x = x_new

    # Check if we converged
    if best_norm <= 1e-8:
        return best_x

    # Method 2: Levenberg-Marquardt from current best
    if time_ok():
        x_lm = best_x.copy()
        for it in range(100):
            if not time_ok():
                break
            Fx = compute_F(x_lm)
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x_lm.copy()
            if norm_Fx <= 1e-10:
                break

            h = h_base * max(1.0, np.linalg.norm(x_lm)) ** 0.5
            J, Fx = finite_diff_jacobian_forward(x_lm, h)

            JtJ = J.T @ J
            lambda_lq = 1e-8
            for _ in range(50):
                try:
                    dx = np.linalg.solve(JtJ + lambda_lq * np.eye(2), -J.T @ Fx)
                except np.linalg.LinAlgError:
                    lambda_lq *= 10
                    continue
                x_trial = x_lm + dx
                Fx_trial = compute_F(x_trial)
                if np.linalg.norm(Fx_trial) < norm_Fx:
                    lambda_lq *= 0.1
                    x_lm = x_trial
                    break
                else:
                    lambda_lq *= 10
                if lambda_lq > 1e15:
                    break

    # Method 3: Random restarts with Newton if still not converged
    if best_norm > 1e-8 and time_ok():
        for attempt in range(10):
            if not time_ok():
                break
            # Try random perturbations of varying magnitudes
            scale = 0.1 * (10 ** (attempt % 3))  # 0.1, 1.0, 10.0
            x_try = best_x + np.random.randn(2) * scale
            x_cur = x_try.copy()
            for it in range(40):
                if not time_ok():
                    break
                Fx = compute_F(x_cur)
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_cur.copy()
                if norm_Fx <= 1e-10:
                    break
                h = h_base * max(1.0, np.linalg.norm(x_cur)) ** 0.5
                J, Fx = finite_diff_jacobian_forward(x_cur, h)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                step_size = 1.0
                x_new = x_cur + step_size * dx
                for _ in range(20):
                    Fx_new = compute_F(x_new)
                    if np.linalg.norm(Fx_new) < norm_Fx:
                        break
                    step_size *= 0.5
                    x_new = x_cur + step_size * dx
                    if step_size < 1e-15:
                        break
                x_cur = x_new

    # Final check: return best_x if it's finite, otherwise return x0
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)
# EVOLVE-BLOCK-END

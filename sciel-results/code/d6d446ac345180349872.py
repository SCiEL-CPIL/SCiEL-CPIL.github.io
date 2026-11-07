"""Root finder for 3 fixed 2-D systems: Broyden-first with polish and re-anchoring."""
import time
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Broyden-first (1 F-call/step), then Newton FD polish, with re-anchoring.
    """
    deadline = time.time() + 145.0
    tol = 1e-8
    best_x = np.array(x0, dtype=float)
    best_norm = float('inf')

    def time_left():
        return deadline - time.time()

    def is_finite(v):
        try:
            a = np.asarray(v, dtype=float)
            return a.shape == (2,) and np.all(np.isfinite(a))
        except Exception:
            return False

    def safe_F(x):
        try:
            fx = np.asarray(F(x), dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return None
            return fx
        except Exception:
            return None

    def update_best(x, Fx):
        nonlocal best_x, best_norm
        if is_finite(x):
            n = np.linalg.norm(Fx)
            if n < best_norm:
                best_norm = n
                best_x = np.array(x, dtype=float)

    def fd_jacobian(x, Fx):
        """Finite-difference Jacobian using 2 extra F-calls."""
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_F(x + e)
            if Fx_plus is None:
                return None
            J[:, j] = (Fx_plus - Fx) / h
        return J

    # Phase 1: Broyden's method - 1 F-call per step
    x = np.array(x0, dtype=float)
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    update_best(x, Fx)
    if np.linalg.norm(Fx) <= tol:
        return x

    # Initialize B with FD Jacobian (3 calls total: Fx + 2 for FD)
    B = fd_jacobian(x, Fx)
    if B is None:
        B = np.eye(2)

    max_broyden_steps = 100
    stall_count = 0
    prev_norm = np.linalg.norm(Fx)

    for step in range(max_broyden_steps):
        if time_left() < 2.0:
            break

        try:
            dx = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx

        if not np.all(np.isfinite(dx)):
            dx = -Fx

        # Limit step size
        dx_norm = np.linalg.norm(dx)
        if dx_norm > 10.0:
            dx = dx * 10.0 / dx_norm

        # Line search with Armijo condition
        alpha = 1.0
        fx_norm = np.linalg.norm(Fx)
        accepted = False
        for _ in range(20):
            x_trial = x + alpha * dx
            Fx_trial = safe_F(x_trial)
            if Fx_trial is None:
                alpha *= 0.5
                continue
            fx_trial_norm = np.linalg.norm(Fx_trial)
            update_best(x_trial, Fx_trial)
            if fx_trial_norm <= tol:
                return x_trial
            if fx_trial_norm < fx_norm * (1 - 1e-4 * alpha):
                # Broyden update
                s = x_trial - x
                y = Fx_trial - Fx
                denom = np.dot(y, s)
                if abs(denom) > 1e-12:
                    B = B + np.outer(y - B @ s, s) / denom
                x = x_trial
                Fx = Fx_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            stall_count += 1
            if stall_count >= 5:
                break
        else:
            stall_count = 0
            curr_norm = np.linalg.norm(Fx)
            if curr_norm >= prev_norm * 0.99:
                # Progress is slow; consider re-anchoring
                stall_count += 1
                if stall_count >= 5:
                    break
            prev_norm = curr_norm

    # Phase 2: If close enough, do Newton FD polish
    if best_norm < 1e-4 and time_left() > 3.0:
        x = np.array(best_x, dtype=float)
        Fx = safe_F(x)
        if Fx is not None:
            for step in range(30):
                if time_left() < 1.0:
                    break
                if Fx is None:
                    break
                update_best(x, Fx)
                if np.linalg.norm(Fx) <= tol:
                    return x
                J = fd_jacobian(x, Fx)
                if J is None:
                    break
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(dx)):
                    break
                alpha = 1.0
                fx_norm = np.linalg.norm(Fx)
                for _ in range(15):
                    x_trial = x + alpha * dx
                    Fx_trial = safe_F(x_trial)
                    if Fx_trial is None:
                        alpha *= 0.5
                        continue
                    fx_trial_norm = np.linalg.norm(Fx_trial)
                    update_best(x_trial, Fx_trial)
                    if fx_trial_norm <= tol:
                        return x_trial
                    if fx_trial_norm < fx_norm * (1 - 1e-4 * alpha):
                        x = x_trial
                        Fx = Fx_trial
                        break
                    alpha *= 0.5

    # Phase 3: If still not converged, try re-anchoring from best_x with fresh Broyden
    if best_norm > tol and time_left() > 5.0:
        x = np.array(best_x, dtype=float)
        Fx = safe_F(x)
        if Fx is not None and np.linalg.norm(Fx) > tol:
            B = fd_jacobian(x, Fx)
            if B is None:
                B = np.eye(2)
            for step in range(50):
                if time_left() < 1.0:
                    break
                try:
                    dx = np.linalg.solve(B, -Fx)
                except np.linalg.LinAlgError:
                    dx = -Fx
                if not np.all(np.isfinite(dx)):
                    dx = -Fx
                dx_norm = np.linalg.norm(dx)
                if dx_norm > 10.0:
                    dx = dx * 10.0 / dx_norm
                alpha = 1.0
                fx_norm = np.linalg.norm(Fx)
                accepted = False
                for _ in range(20):
                    x_trial = x + alpha * dx
                    Fx_trial = safe_F(x_trial)
                    if Fx_trial is None:
                        alpha *= 0.5
                        continue
                    fx_trial_norm = np.linalg.norm(Fx_trial)
                    update_best(x_trial, Fx_trial)
                    if fx_trial_norm <= tol:
                        return x_trial
                    if fx_trial_norm < fx_norm * (1 - 1e-4 * alpha):
                        s = x_trial - x
                        y = Fx_trial - Fx
                        denom = np.dot(y, s)
                        if abs(denom) > 1e-12:
                            B = B + np.outer(y - B @ s, s) / denom
                        x = x_trial
                        Fx = Fx_trial
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    break

    return best_x
# EVOLVE-BLOCK-END

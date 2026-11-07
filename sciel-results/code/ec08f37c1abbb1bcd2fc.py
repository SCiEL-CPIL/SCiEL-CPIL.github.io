"""Root finder for 3 fixed 2-D systems: Broyden-first with trust-region fallback."""
import time
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method (1 F-call/iter after setup) with line search,
    then trust-region damped Newton, then grid refinement as fallbacks.
    """
    deadline = time.time() + 145.0  # safety margin before 150s kill
    tol = 1e-8
    best_x = np.array(x0, dtype=float)
    best_norm = float('inf')
    call_count = 0

    def time_left():
        return deadline - time.time()

    def safe_F(x):
        nonlocal call_count
        try:
            fx = np.asarray(F(x), dtype=float)
            call_count += 1
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return None
            return fx
        except Exception:
            call_count += 1
            return None

    def update_best(x, Fx):
        nonlocal best_x, best_norm
        try:
            a = np.asarray(x, dtype=float)
            if a.shape == (2,) and np.all(np.isfinite(a)):
                n = np.linalg.norm(Fx)
                if n < best_norm:
                    best_norm = n
                    best_x = a.copy()
        except Exception:
            pass

    def norm(v):
        return np.linalg.norm(v)

    # === Strategy 1: Broyden's method with line search ===
    x = np.array(x0, dtype=float)
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    update_best(x, Fx)
    if norm(Fx) <= tol:
        return x

    # Initialize Broyden matrix with finite-difference Jacobian (3 calls: x, x+e0, x+e1)
    h = 1e-7
    B = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_p = safe_F(x + e)
        if Fx_p is None:
            B = np.eye(2)
            break
        B[:, j] = (Fx_p - Fx) / h
    else:
        pass

    max_broyden = 450  # calls remaining ~ 500 - 3 = 497, use 450 for safety
    for step in range(max_broyden):
        if time_left() < 1.0:
            break
        if call_count >= 495:
            break

        Fx_norm = norm(Fx)
        if Fx_norm <= tol:
            return x

        # Broyden step
        try:
            dx = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx

        if not np.all(np.isfinite(dx)):
            dx = -Fx

        # Backtracking line search
        alpha = 1.0
        accepted = False
        for _ in range(25):
            x_trial = x + alpha * dx
            Fx_trial = safe_F(x_trial)
            if Fx_trial is None:
                alpha *= 0.5
                continue
            fx_trial_norm = norm(Fx_trial)
            update_best(x_trial, Fx_trial)
            if fx_trial_norm <= tol:
                return x_trial
            # Armijo-like condition
            if fx_trial_norm < Fx_norm * (1 - 1e-4 * alpha):
                s = x_trial - x
                y = Fx_trial - Fx
                denom = np.dot(y, s)
                if abs(denom) > 1e-14:
                    B = B + np.outer(y - B @ s, s) / denom
                x = x_trial
                Fx = Fx_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # Try damped step: x - Fx (steepest descent on ||F||^2)
            dx2 = -Fx
            alpha2 = 1.0
            for _ in range(15):
                x_trial = x + alpha2 * dx2
                Fx_trial = safe_F(x_trial)
                if Fx_trial is None:
                    alpha2 *= 0.5
                    continue
                fx_trial_norm = norm(Fx_trial)
                update_best(x_trial, Fx_trial)
                if fx_trial_norm <= tol:
                    return x_trial
                if fx_trial_norm < Fx_norm * 0.99:
                    s = x_trial - x
                    y = Fx_trial - Fx
                    denom = np.dot(y, s)
                    if abs(denom) > 1e-14:
                        B = B + np.outer(y - B @ s, s) / denom
                    x = x_trial
                    Fx = Fx_trial
                    accepted = True
                    break
                alpha2 *= 0.5
            if not accepted:
                break

        if norm(Fx) <= tol:
            return x

    if best_norm <= tol:
        return best_x

    # === Strategy 2: Trust-region damped Newton (if calls remain) ===
    if call_count < 490 and time_left() > 2.0:
        x = np.array(x0, dtype=float)
        Fx = safe_F(x)
        if Fx is None:
            return best_x
        update_best(x, Fx)
        if norm(Fx) <= tol:
            return x

        # Compute FD Jacobian
        h = 1e-6
        J = np.zeros((2, 2))
        ok = True
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_F(x + e)
            if Fx_p is None:
                ok = False
                break
            J[:, j] = (Fx_p - Fx) / h

        if ok:
            max_tr = 20
            for step in range(max_tr):
                if time_left() < 1.0 or call_count >= 495:
                    break
                Fx_norm = norm(Fx)
                if Fx_norm <= tol:
                    return x
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(dx)):
                    break

                # Damped Newton with backtracking
                alpha = 1.0
                accepted = False
                for _ in range(20):
                    x_trial = x + alpha * dx
                    Fx_trial = safe_F(x_trial)
                    if Fx_trial is None:
                        alpha *= 0.5
                        continue
                    fx_trial_norm = norm(Fx_trial)
                    update_best(x_trial, Fx_trial)
                    if fx_trial_norm <= tol:
                        return x_trial
                    if fx_trial_norm < Fx_norm * (1 - 1e-4 * alpha):
                        # Update Jacobian with FD
                        h2 = 1e-6
                        J_new = np.zeros((2, 2))
                        for j in range(2):
                            e = np.zeros(2)
                            e[j] = h2
                            Fx_p = safe_F(x_trial + e)
                            if Fx_p is None:
                                break
                            J_new[:, j] = (Fx_p - Fx_trial) / h2
                        else:
                            J = J_new
                        x = x_trial
                        Fx = Fx_trial
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    break

    # === Strategy 3: Coarse grid + refinement (last resort, very few calls) ===
    if call_count < 495 and time_left() > 2.0 and best_norm > 1e-4:
        # Try a small set of points around x0
        x0_arr = np.array(x0, dtype=float)
        for scale in [0.1, 0.5, 1.0, 5.0, 10.0]:
            if call_count >= 495 or time_left() < 1.0:
                break
            for dx1 in [-scale, 0.0, scale]:
                for dx2 in [-scale, 0.0, scale]:
                    if call_count >= 495 or time_left() < 1.0:
                        break
                    x_try = x0_arr + np.array([dx1, dx2])
                    Fx_try = safe_F(x_try)
                    if Fx_try is not None:
                        update_best(x_try, Fx_try)
                        if norm(Fx_try) <= tol:
                            return x_try

    return best_x

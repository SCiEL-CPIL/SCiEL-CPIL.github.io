"""Root finder for 3 fixed 2-D systems: robust quasi-Newton with safeguards."""
import time
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a robust quasi-Newton (Broyden) method with line search,
    finite-difference Jacobian fallback, and multiple restarts.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    tol = 1e-8
    best_x = np.array(x0, dtype=float)
    best_norm = np.inf

    def time_left():
        return deadline - time.time()

    def is_finite(v):
        try:
            a = np.asarray(v, dtype=float)
            return a.shape == (2,) and np.all(np.isfinite(a))
        except Exception:
            return False

    def safe_F(x):
        """Call F, return (value, ok)."""
        try:
            v = F(x)
            v = np.asarray(v, dtype=float)
            if v.shape == (2,) and np.all(np.isfinite(v)):
                return v, True
            return None, False
        except Exception:
            return None, False

    def update_best(x, Fx):
        nonlocal best_x, best_norm
        n = np.linalg.norm(Fx)
        if n < best_norm:
            best_norm = n
            best_x = x.copy()

    # --- Broyden's method with backtracking line search ---
    def broyden(F, x0, max_iter=100):
        x = np.array(x0, dtype=float)
        Fx, ok = safe_F(x)
        if not ok:
            return x, np.inf
        update_best(x, Fx)
        if np.linalg.norm(Fx) <= tol:
            return x, np.linalg.norm(Fx)

        # Initial Jacobian: finite differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e, ok_e = safe_F(x + e)
            if not ok_e:
                # Use identity if FD fails
                J = np.eye(2)
                break
            J[:, j] = (Fx_e - Fx) / h
        else:
            pass

        for it in range(max_iter):
            if time_left() < 2.0:
                break
            if np.linalg.norm(Fx) <= tol:
                return x, np.linalg.norm(Fx)

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Reset Jacobian to identity and try again
                J = np.eye(2)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break

            # Ensure dx is finite
            if not np.all(np.isfinite(dx)):
                break

            # Backtracking line search
            alpha = 1.0
            x_new = x
            Fx_new = Fx
            for _ in range(30):
                x_trial = x + alpha * dx
                Fx_trial, ok_t = safe_F(x_trial)
                if not ok_t:
                    alpha *= 0.5
                    continue
                n_trial = np.linalg.norm(Fx_trial)
                if n_trial < np.linalg.norm(Fx):
                    x_new = x_trial
                    Fx_new = Fx_trial
                    break
                alpha *= 0.5
            else:
                # No improvement found; try a small step
                x_new = x + 0.01 * dx
                Fx_new, ok_t = safe_F(x_new)
                if not ok_t:
                    break

            # Broyden update
            s = x_new - x
            y = Fx_new - Fx
            if np.linalg.norm(s) > 1e-15:
                J = J + (y - J @ s) @ s.T / (s.T @ s)

            x = x_new
            Fx = Fx_new
            update_best(x, Fx)

        return x, np.linalg.norm(Fx)

    # --- Newton with FD Jacobian and damping ---
    def newton_fd(F, x0, max_iter=80):
        x = np.array(x0, dtype=float)
        Fx, ok = safe_F(x)
        if not ok:
            return x, np.inf
        update_best(x, Fx)
        if np.linalg.norm(Fx) <= tol:
            return x, np.linalg.norm(Fx)

        h = 1e-6
        for it in range(max_iter):
            if time_left() < 2.0:
                break
            if np.linalg.norm(Fx) <= tol:
                return x, np.linalg.norm(Fx)

            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e, ok_e = safe_F(x + e)
                if not ok_e:
                    break
                J[:, j] = (Fx_e - Fx) / h
            else:
                try:
                    dx = np.linalg.solve(J, -Fx)
                    if not np.all(np.isfinite(dx)):
                        break
                    # Damped Newton with backtracking
                    alpha = 1.0
                    x_new = x
                    Fx_new = Fx
                    for _ in range(30):
                        x_trial = x + alpha * dx
                        Fx_trial, ok_t = safe_F(x_trial)
                        if not ok_t:
                            alpha *= 0.5
                            continue
                        if np.linalg.norm(Fx_trial) < np.linalg.norm(Fx):
                            x_new = x_trial
                            Fx_new = Fx_trial
                            break
                        alpha *= 0.5
                    x = x_new
                    Fx = Fx_new
                    update_best(x, Fx)
                except np.linalg.LinAlgError:
                    break
            # If FD failed, break
            if not ok_e:
                break

        return x, np.linalg.norm(Fx)

    # --- Nelder-Mead on ||F||^2 ---
    def nelder_mead(F, x0, max_iter=200):
        x = np.array(x0, dtype=float)
        Fx, ok = safe_F(x)
        if not ok:
            return x, np.inf
        update_best(x, Fx)
        if np.linalg.norm(Fx) <= tol:
            return x, np.linalg.norm(Fx)

        # Initial simplex
        scale = max(1.0, np.linalg.norm(x))
        simplex = [
            x.copy(),
            x + np.array([scale * 0.1, 0.0]),
            x + np.array([0.0, scale * 0.1]),
        ]
        fvals = []
        for v in simplex:
            Fv, ok_v = safe_F(v)
            if ok_v:
                fvals.append(np.linalg.norm(Fv) ** 2)
            else:
                fvals.append(1e10)

        for it in range(max_iter):
            if time_left() < 2.0:
                break
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if fvals[0] <= tol ** 2:
                return simplex[0], np.sqrt(fvals[0])

            # Centroid of all but worst
            centroid = (simplex[0] + simplex[1]) / 2.0

            # Reflection
            xr = centroid + (centroid - simplex[2])
            Fxr, ok_r = safe_F(xr)
            fr = np.linalg.norm(Fxr) ** 2 if ok_r else 1e10

            if fr < fvals[1]:
                # Expansion
                xe = centroid + 2.0 * (centroid - simplex[2])
                Fxe, ok_e = safe_F(xe)
                fe = np.linalg.norm(Fxe) ** 2 if ok_e else 1e10
                if fe < fr:
                    simplex[2] = xe
                    fvals[2] = fe
                else:
                    simplex[2] = xr
                    fvals[2] = fr
            elif fr < fvals[0]:
                simplex[2] = xr
                fvals[2] = fr
            else:
                # Contraction
                xc = centroid + 0.5 * (simplex[2] - centroid)
                Fxc, ok_c = safe_F(xc)
                fc = np.linalg.norm(Fxc) ** 2 if ok_c else 1e10
                if fc < fvals[2]:
                    simplex[2] = xc
                    fvals[2] = fc
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        Fsi, ok_s = safe_F(simplex[i])
                        fvals[i] = np.linalg.norm(Fsi) ** 2 if ok_s else 1e10

            # Track best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_norm ** 2:
                best_norm = np.sqrt(fvals[min_idx])
                best_x = simplex[min_idx].copy()

        return best_x, best_norm

    # --- Run multiple strategies ---
    strategies = [
        ("broyden", broyden),
        ("newton_fd", newton_fd),
        ("nelder_mead", nelder_mead),
    ]

    for name, method in strategies:
        if time_left() < 5.0:
            break
        x_result, norm_result = method(F, x0)
        if is_finite(x_result) and np.linalg.norm(F(x_result) if is_finite(F(x_result)) else np.inf) <= tol:
            return x_result
        # Update best
        Fx_res, ok_res = safe_F(x_result)
        if ok_res:
            update_best(x_result, Fx_res)
        if best_norm <= tol:
            return best_x

    # If still not converged, try a few random restarts with Broyden
    if time_left() > 10.0:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time_left() < 5.0:
                break
            x_start = best_x + rng.normal(0, 0.5, size=2)
            x_result, norm_result = broyden(F, x_start, max_iter=50)
            Fx_res, ok_res = safe_F(x_result)
            if ok_res:
                update_best(x_result, Fx_res)
            if best_norm <= tol:
                return best_x

    return best_x

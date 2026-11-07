"""Root finder for 3 fixed 2D systems using trust-region Newton with homotopy continuation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses trust-region Newton as primary, homotopy continuation as fallback.
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)

    start_time = time.time()
    deadline = start_time + 145.0

    def time_up():
        return time.time() > deadline

    # Try to evaluate F at x0
    try:
        Fx = F(x)
    except Exception:
        Fx = None

    if Fx is not None and not np.all(np.isfinite(Fx)):
        # Try perturbations
        found = False
        for scale in [1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0]:
            for dx, dy in [(scale, 0), (-scale, 0), (0, scale), (0, -scale),
                           (scale, scale), (scale, -scale), (-scale, scale), (-scale, -scale)]:
                if time_up():
                    break
                x_try = np.array([x[0] + dx, x[1] + dy])
                try:
                    Fx_try = F(x_try)
                except Exception:
                    continue
                if np.all(np.isfinite(Fx_try)):
                    x = x_try
                    Fx = Fx_try
                    found = True
                    break
            if found or time_up():
                break
        if not found:
            # Go to homotopy
            return _homotopy_primary(F, x, start_time, deadline)

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # --- Phase 1: Trust-region Newton from x0 ---
    result = _trust_region_newton(F, x, Fx, start_time, deadline)
    if result is not None:
        return result

    # --- Phase 2: Homotopy continuation ---
    result = _homotopy_primary(F, x, start_time, deadline)
    if result is not None:
        return result

    # --- Phase 3: Grid search + Newton ---
    result = _grid_search(F, x, start_time, deadline)
    if result is not None:
        return result

    # Return best we have
    return x


def _trust_region_newton(F, x0, Fx0, start_time, deadline):
    """Trust-region Newton method with adaptive step size."""
    def time_up():
        return time.time() > deadline

    x = np.array(x0, dtype=float).copy()
    Fx = np.array(Fx0, dtype=float).copy()
    norm_Fx = np.linalg.norm(Fx)
    best_x = x.copy()
    best_norm = norm_Fx

    # Initial Jacobian via central differences
    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fp = F(x + e)
            Fm = F(x - e)
            if np.all(np.isfinite(Fp)) and np.all(np.isfinite(Fm)):
                J[:, j] = (Fp - Fm) / (2 * h)
            else:
                Fp = F(x + e)
                if np.all(np.isfinite(Fp)):
                    J[:, j] = (Fp - Fx) / h
                else:
                    J[:, j] = 0.0
        except Exception:
            J[:, j] = 0.0

    delta = 1.0  # Trust region radius
    for iteration in range(40):
        if time_up():
            break

        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                d = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(d)):
            break

        norm_d = np.linalg.norm(d)
        if norm_d < 1e-14:
            break

        # Scale step to trust region
        if norm_d > delta:
            d = d * (delta / norm_d)

        # Try full step
        x_new = x + d
        try:
            Fx_new = F(x_new)
        except Exception:
            delta *= 0.5
            continue

        if not np.all(np.isfinite(Fx_new)):
            delta *= 0.5
            continue

        norm_new = np.linalg.norm(Fx_new)

        if norm_new <= 1e-8:
            return x_new

        # Check if step is useful
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new.copy()

        # Accept if significant decrease
        if norm_new < (1.0 - 1e-4 * np.linalg.norm(d)) * norm_Fx:
            # Good variant Broyden update
            d_dot_d = np.dot(d, d)
            if d_dot_d > 1e-16:
                delta_F = Fx_new - Fx - J @ d
                J = J + np.outer(delta_F, d) / d_dot_d

            x = x_new
            Fx = Fx_new
            norm_Fx = norm_new
            delta = min(delta * 2.0, 10.0)
        else:
            delta *= 0.5

        # Convergence check
        if norm_Fx < 1e-10 and iteration >= 3:
            break

    if best_norm <= 1e-8:
        return best_x
    if best_norm < 1e-6:
        return best_x
    return None


def _homotopy_primary(F, x_start, start_time, deadline):
    """
    Homotopy continuation: H_t(x) = (1-t)*(x - anchor) + t*F(x)
    Trace from t=0 (known root at anchor) to t=1 (root of F).
    """
    def time_up():
        return time.time() > deadline

    best_x = np.array(x_start, dtype=float).copy()
    best_norm = np.inf
    try:
        Fx_best = F(best_x)
        if np.all(np.isfinite(Fx_best)):
            best_norm = np.linalg.norm(Fx_best)
    except Exception:
        pass

    # Try multiple anchors
    anchors = [
        np.array(x_start, dtype=float),
        x_start + np.array([1.0, 0.0]),
        x_start + np.array([0.0, 1.0]),
        x_start + np.array([-1.0, 0.0]),
        x_start + np.array([0.0, -1.0]),
        x_start + np.array([1.0, 1.0]),
        x_start + np.array([-1.0, -1.0]),
        x_start + np.array([0.5, 0.5]),
    ]

    for anchor in anchors:
        if time_up():
            break

        x_curr = anchor.copy()
        t_curr = 0.0
        dt = 0.15

        for t_step in range(20):
            if time_up():
                break

            t_target = min(t_curr + dt, 1.0)

            def H_t(x_val, t_val=t_target):
                return (1 - t_val) * (x_val - anchor) + t_val * F(x_val)

            # Solve H_t(x) = 0 from x_curr using Newton
            x_sol, converged = _newton_solve(H_t, x_curr, start_time, deadline, tol=1e-10)

            if converged:
                if t_target >= 0.999:
                    # Check actual F
                    try:
                        Fx_check = F(x_sol)
                        if np.all(np.isfinite(Fx_check)):
                            norm_check = np.linalg.norm(Fx_check)
                            if norm_check <= 1e-8:
                                return x_sol
                            if norm_check < best_norm:
                                best_norm = norm_check
                                best_x = x_sol.copy()
                    except Exception:
                        pass
                    break
                else:
                    x_curr = x_sol
                    t_curr = t_target
                    # Increase dt if things went well
                    dt = min(dt * 1.3, 0.4)
            else:
                dt = max(dt * 0.5, 0.03)

    # Refine best result
    if best_norm < np.inf and not time_up():
        result = _trust_region_newton(F, best_x, None, start_time, deadline)
        if result is not None:
            return result
        # One more Newton pass
        try:
            Fx_b = F(best_x)
            if np.all(np.isfinite(Fx_b)):
                result = _trust_region_newton(F, best_x, Fx_b, start_time, deadline)
                if result is not None:
                    return result
        except Exception:
            pass

    if best_norm < np.inf:
        return best_x
    return None


def _newton_solve(func, x0, start_time, deadline, tol=1e-10, max_iter=25):
    """Simple Newton solver for a 2D system."""
    def time_up():
        return time.time() > deadline

    x = np.array(x0, dtype=float).copy()
    try:
        Fx = func(x)
    except Exception:
        return x, False

    if not np.all(np.isfinite(Fx)):
        return x, False

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= tol:
        return x, True

    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fp = func(x + e)
            Fm = func(x - e)
            if np.all(np.isfinite(Fp)) and np.all(np.isfinite(Fm)):
                J[:, j] = (Fp - Fm) / (2 * h)
            else:
                Fp = func(x + e)
                if np.all(np.isfinite(Fp)):
                    J[:, j] = (Fp - Fx) / h
                else:
                    J[:, j] = 0.0
        except Exception:
            J[:, j] = 0.0

    for it in range(max_iter):
        if time_up():
            return x, False

        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                d = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                return x, False

        if not np.all(np.isfinite(d)):
            return x, False
        if np.linalg.norm(d) < 1e-14:
            return x, norm_Fx <= tol

        alpha = 1.0
        accepted = False
        for _ in range(10):
            if time_up():
                return x, False
            x_new = x + alpha * d
            try:
                Fx_new = func(x_new)
            except Exception:
                alpha *= 0.5
                continue
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            norm_new = np.linalg.norm(Fx_new)
            if norm_new <= tol:
                return x_new, True
            if norm_new < (1.0 - 1e-4 * alpha) * norm_Fx:
                d_dot_d = np.dot(d, d)
                if d_dot_d > 1e-16:
                    delta_F = Fx_new - Fx - J @ d
                    J = J + np.outer(delta_F, d) / d_dot_d
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_new
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return x, False

        if norm_Fx <= tol:
            return x, True

    return x, norm_Fx <= tol


def _grid_search(F, x0, start_time, deadline):
    """Coarse grid search around x0, then Newton refinement on best points."""
    def time_up():
        return time.time() > deadline

    best_x = np.array(x0, dtype=float).copy()
    best_norm = np.inf
    try:
        Fx = F(best_x)
        if np.all(np.isfinite(Fx)):
            best_norm = np.linalg.norm(Fx)
    except Exception:
        pass

    # Grid of offsets
    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    candidates = []
    for s in scales:
        for dx in [-s, -s/2, 0, s/2, s]:
            for dy in [-s, -s/2, 0, s/2, s]:
                candidates.append(x0 + np.array([dx, dy]))

    # Evaluate all candidates
    for xc in candidates:
        if time_up():
            break
        try:
            Fx = F(xc)
        except Exception:
            continue
        if not np.all(np.isfinite(Fx)):
            continue
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_norm = norm
            best_x = xc.copy()

    if best_norm <= 1e-8:
        return best_x

    # Newton refinement from best point
    if not time_up():
        result = _trust_region_newton(F, best_x, None, start_time, deadline)
        if result is not None:
            return result

    # Also try a few other good candidates
    for xc in candidates[:10]:
        if time_up():
            break
        try:
            Fx = F(xc)
        except Exception:
            continue
        if not np.all(np.isfinite(Fx)):
            continue
        if np.linalg.norm(Fx) < best_norm * 0.5:
            result = _trust_region_newton(F, xc, Fx, start_time, deadline)
            if result is not None:
                return result

    return best_x if best_norm < np.inf else None


# EVOLVE-BLOCK-END

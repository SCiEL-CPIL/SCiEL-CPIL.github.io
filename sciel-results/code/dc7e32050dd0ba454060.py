"""Root finder for 3 fixed two-dimensional systems using robust Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's method with line search.
    
    Strategy:
    1. Evaluate F at x0 for baseline.
    2. Compute initial Jacobian via forward differences (2 calls).
    3. Run Broyden's method with backtracking line search (1 call/iteration).
    4. If stuck, try random restarts with Broyden from perturbed points.
    
    Key insight: Broyden uses 1 F-call per iteration (vs 3-5 for Newton),
    enabling more iterations in the same call budget and better escape from
    local minima/singularities.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s margin before 150s limit

    def time_ok():
        return time.time() < deadline

    x = np.array(x0, dtype=float).copy()
    n = 2

    best_x = x.copy()
    best_norm = np.inf

    # Initial evaluation
    try:
        Fx = np.asarray(F(x), dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return x
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        best_norm = norm_Fx
    except Exception:
        return x

    def compute_F(val):
        val = np.asarray(val, dtype=float)
        result = np.asarray(F(val), dtype=float)
        return result

    def broyden_solve(x_start, Fx_start, max_iter=80):
        """Run Broyden's method from x_start with initial Fx_start."""
        nonlocal best_x, best_norm
        if not time_ok():
            return None

        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_cur = np.linalg.norm(Fx_cur)

        # Initial Jacobian via forward differences (2 calls)
        h = 1e-6 * max(1.0, np.linalg.norm(x_cur))
        J = np.zeros((2, 2))
        try:
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_pert = compute_F(x_cur + e)
                if Fx_pert.shape != (2,) or not np.all(np.isfinite(Fx_pert)):
                    return None
                J[:, j] = (Fx_pert - Fx_cur) / h
        except Exception:
            return None

        if not np.all(np.isfinite(J)):
            return None

        # Broyden iterations
        for it in range(max_iter):
            if not time_ok():
                break

            if norm_cur <= 1e-10:
                break

            # Solve J @ s = -Fx
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                # Regularize
                J_reg = J + 1e-8 * np.eye(2)
                try:
                    s = np.linalg.solve(J_reg, -Fx_cur)
                except np.linalg.LinAlgError:
                    break

            if not np.all(np.isfinite(s)):
                break

            # Backtracking line search
            alpha = 1.0
            accepted = False
            x_new = x_cur
            Fx_new = Fx_cur
            norm_new = norm_cur

            for _ in range(25):
                x_trial = x_cur + alpha * s
                try:
                    Fx_trial = compute_F(x_trial)
                    if Fx_trial.shape != (2,) or not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    norm_trial = np.linalg.norm(Fx_trial)
                except Exception:
                    alpha *= 0.5
                    continue

                if norm_trial < norm_cur:
                    x_new = x_trial
                    Fx_new = Fx_trial
                    norm_new = norm_trial
                    accepted = True
                    break
                alpha *= 0.5
                if alpha < 1e-12:
                    break

            if not accepted:
                break

            # Update Broyden Jacobian
            s_actual = x_new - x_cur
            y = Fx_new - Fx_cur
            denom = np.dot(s_actual, s_actual)
            if denom > 1e-14:
                J = J + np.outer(y - J @ s_actual, s_actual) / denom
            else:
                # Recompute Jacobian if step too small
                h_re = 1e-6 * max(1.0, np.linalg.norm(x_new))
                J = np.zeros((2, 2))
                try:
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h_re
                        Fx_pert = compute_F(x_new + e)
                        J[:, j] = (Fx_pert - Fx_new) / h_re
                except Exception:
                    break

            x_cur = x_new
            Fx_cur = Fx_new
            norm_cur = norm_new

            if norm_cur < best_norm:
                best_norm = norm_cur
                best_x = x_cur.copy()

            if norm_cur <= 1e-8:
                return x_cur

        return x_cur

    # Phase 1: Broyden from x0
    result = broyden_solve(x, Fx, max_iter=100)
    if result is not None and best_norm <= 1e-8:
        return best_x

    # Phase 2: If not converged, try random restarts with Broyden
    if best_norm > 1e-8 and time_ok():
        np.random.seed(42)
        for attempt in range(20):
            if not time_ok():
                break

            # Perturb from best_x with varying scales
            scale = 0.2 * (attempt + 1)
            x_try = best_x + np.random.randn(2) * scale

            try:
                Fx_try = compute_F(x_try)
                if Fx_try.shape != (2,) or not np.all(np.isfinite(Fx_try)):
                    continue
                norm_try = np.linalg.norm(Fx_try)
            except Exception:
                continue

            if norm_try < best_norm:
                best_norm = norm_try
                best_x = x_try.copy()

            if norm_try <= 1e-8:
                return best_x

            # Run Broyden from this point
            result = broyden_solve(x_try, Fx_try, max_iter=50)
            if result is not None and best_norm <= 1e-8:
                return best_x

    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)

# EVOLVE-BLOCK-END

"""Root finder for 3 fixed 2D systems: Broyden with high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Use Broyden's method with rank-1 Jacobian updates (minimizes F-calls).
    2. Backtracking line search on ||F||^2.
    3. Final high-precision polish pass with exact Newton steps.
    4. Multi-start with diverse initial points if needed.
    """
    x0_arr = np.asarray(x0, dtype=float).copy()
    if x0_arr.shape != (2,) or not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)

    tol = 1e-8
    tol_sq = tol * tol
    t0 = time.time()
    deadline = 148.0

    def remaining():
        return deadline - (time.time() - t0)

    def safe_F(x_pt):
        """Call F with finite guard; return None on failure."""
        try:
            y = np.asarray(F(x_pt), dtype=float)
            if y.shape != (2,) or not np.all(np.isfinite(y)):
                return None
            return y
        except Exception:
            return None

    def compute_jacobian(x, Fx):
        """Compute Jacobian via forward differences. Uses 2 F-calls."""
        h = 1e-6
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx_e1 = safe_F(x + e1)
        Fx_e2 = safe_F(x + e2)
        if Fx_e1 is None or Fx_e2 is None:
            return None
        J = np.column_stack([(Fx_e1 - Fx) / h, (Fx_e2 - Fx) / h])
        return J

    def broyden_solve(x_start, Fx_start, max_iters=25):
        """
        Broyden's method with rank-1 Jacobian updates.
        Returns (x, Fx, converged, calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0

        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, calls_used

        # Initial Jacobian estimate
        J = compute_jacobian(x_cur, Fx_cur)
        if J is None:
            return x_cur, Fx_cur, False, 2
        calls_used += 2

        for _iter in range(max_iters):
            if remaining() < 1.0:
                break

            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break

            # Backtracking line search
            alpha = 1.0
            accepted = False
            x_try = x_cur
            Fx_try = Fx_cur
            norm_sq_try = norm_sq_cur

            for _ls in range(8):
                x_try = x_cur + alpha * s
                Fx_try = safe_F(x_try)
                calls_used += 1
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, calls_used
                if norm_sq_try < norm_sq_cur * 0.9:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                break

            # Broyden rank-1 update
            s_actual = x_try - x_cur
            F_diff = Fx_try - Fx_cur
            denom = np.dot(s_actual, s_actual)
            if denom > 1e-30:
                J = J + np.outer(F_diff - J @ s_actual, s_actual) / denom

            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try

        return x_cur, Fx_cur, norm_sq_cur <= tol_sq, calls_used

    def polish_newton(x_start, Fx_start, max_iters=10):
        """
        High-precision Newton polish with recomputed Jacobians.
        Returns (x, Fx, converged, calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0

        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, calls_used

        for _iter in range(max_iters):
            if remaining() < 0.5:
                break

            J = compute_jacobian(x_cur, Fx_cur)
            if J is None:
                break
            calls_used += 2

            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break

            x_try = x_cur + s
            Fx_try = safe_F(x_try)
            calls_used += 1
            if Fx_try is None:
                break

            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            if norm_sq_try <= tol_sq:
                return x_try, Fx_try, True, calls_used

            if norm_sq_try < norm_sq_cur:
                x_cur = x_try
                Fx_cur = Fx_try
                norm_sq_cur = norm_sq_try
            else:
                break

        return x_cur, Fx_cur, norm_sq_cur <= tol_sq, calls_used

    # Main solver
    best_x = x0_arr.copy()
    best_norm_sq = float('inf')

    # Initial evaluation
    Fx0 = safe_F(x0_arr)
    if Fx0 is None:
        return np.zeros(2)
    best_norm_sq = float(np.dot(Fx0, Fx0))
    if best_norm_sq <= tol_sq:
        return x0_arr

    # Primary attempt from x0 with Broyden
    x_sol, Fx_sol, conv, _ = broyden_solve(x0_arr, Fx0, max_iters=25)
    ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
    if ns < best_norm_sq:
        best_norm_sq = ns
        best_x = x_sol.copy()

    # Polish if we got close
    if best_norm_sq < 1e-4:
        x_pol, Fx_pol, conv_pol, _ = polish_newton(best_x, Fx_sol, max_iters=10)
        ns_pol = float(np.dot(Fx_pol, Fx_pol)) if Fx_pol is not None else float('inf')
        if ns_pol < best_norm_sq:
            best_norm_sq = ns_pol
            best_x = x_pol.copy()

    if best_norm_sq <= tol_sq:
        return best_x

    # Multi-start
    rng = np.random.RandomState(12345)
    max_restarts = 12
    for restart in range(max_restarts):
        if remaining() < 2.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x

        # Generate diverse starting points
        if restart % 4 == 0:
            x_start = x0_arr + rng.uniform(-5, 5, size=2)
        elif restart % 4 == 1:
            radius = 1.0 + restart * 0.5
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        elif restart % 4 == 2:
            x_start = x0_arr * rng.uniform(0.5, 3.0)
        else:
            x_start = x0_arr + rng.uniform(-20, 20, size=2)

        if not np.all(np.isfinite(x_start)):
            continue

        Fx_start = safe_F(x_start)
        if Fx_start is None:
            continue

        ns_start = float(np.dot(Fx_start, Fx_start))
        if ns_start < best_norm_sq:
            best_norm_sq = ns_start
            best_x = x_start.copy()
            if best_norm_sq <= tol_sq:
                return best_x

        # Broyden from this start
        x_sol, Fx_sol, conv, _ = broyden_solve(x_start, Fx_start, max_iters=20)
        ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
        if ns < best_norm_sq:
            best_norm_sq = ns
            best_x = x_sol.copy()

        # Polish if close
        if best_norm_sq < 1e-4:
            x_pol, Fx_pol, conv_pol, _ = polish_newton(best_x, Fx_sol, max_iters=8)
            ns_pol = float(np.dot(Fx_pol, Fx_pol)) if Fx_pol is not None else float('inf')
            if ns_pol < best_norm_sq:
                best_norm_sq = ns_pol
                best_x = x_pol.copy()

        if best_norm_sq <= tol_sq:
            return best_x

    return best_x
# EVOLVE-BLOCK-END

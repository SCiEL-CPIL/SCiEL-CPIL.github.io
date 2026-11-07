import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0

    def time_ok():
        return time.time() - start_time < time_limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    call_count = 0

    def F_count(x_pt):
        nonlocal call_count
        call_count += 1
        return F(x_pt)

    # Initial evaluation
    try:
        Fx = F_count(x)
    except:
        return np.zeros(2)

    if not np.all(np.isfinite(Fx)):
        return x

    norm_sq = float(np.dot(Fx, Fx))
    if norm_sq <= tol_sq:
        return x

    best_x = x.copy()
    best_norm_sq = norm_sq

    def compute_jacobian_fd(x_pt, Fx_pt):
        """Compute Jacobian via forward finite differences (2 calls)."""
        h = 1e-7
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt + e2)
            J = np.array([(Fx1 - Fx_pt) / h, (Fx2 - Fx_pt) / h]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    # Initial Jacobian
    J = compute_jacobian_fd(x, Fx)
    if J is None:
        J = np.eye(2)

    # Broyden's method with adaptive step control
    max_iter = 150
    stagnation_count = 0
    lambda_lm = 1e-3

    for _ in range(max_iter):
        if not time_ok() or call_count > 480:
            break

        if norm_sq <= tol_sq:
            return x

        # LM-regularized Newton step
        try:
            JtJ = J.T @ J
            JtF = J.T @ Fx
            A = JtJ + lambda_lm * np.eye(2)
            s = np.linalg.solve(A, -JtF)
        except np.linalg.LinAlgError:
            lambda_lm *= 10.0
            continue

        # Adaptive step size control
        s_norm = np.linalg.norm(s)
        if s_norm > 5.0:
            s = s * (5.0 / s_norm)
            s_norm = 5.0

        x_try = x + s
        try:
            Fx_try = F_count(x_try)
        except:
            lambda_lm *= 10.0
            continue

        if not np.all(np.isfinite(Fx_try)):
            lambda_lm *= 10.0
            continue

        norm_sq_try = float(np.dot(Fx_try, Fx_try))

        if norm_sq_try < norm_sq:
            # Accept step
            dx = x_try - x
            dF = Fx_try - Fx

            # Broyden rank-1 update: J_new = J + (dF - J*dx) * dx^T / (dx^T * dx)
            Jdx = J @ dx
            residual = dF - Jdx
            dx_norm_sq = float(np.dot(dx, dx))
            if dx_norm_sq > 1e-30:
                J = J + np.outer(residual, dx) / dx_norm_sq

            x = x_try
            Fx = Fx_try
            norm_sq = norm_sq_try
            lambda_lm = max(1e-12, lambda_lm * 0.3)
            stagnation_count = 0

            if norm_sq < best_norm_sq:
                best_norm_sq = norm_sq
                best_x = x.copy()

            # Periodically recompute Jacobian to avoid drift
            if _ % 10 == 0:
                J_new = compute_jacobian_fd(x, Fx)
                if J_new is not None:
                    J = J_new
        else:
            # Reject step
            lambda_lm *= 10.0
            stagnation_count += 1

            if stagnation_count > 8:
                break

    # If primary method didn't converge, try Nelder-Mead style direct search
    # to find a better basin, then switch back to Broyden
    if norm_sq > tol_sq and call_count < 440 and time_ok():
        # Simple direct search: try perturbations around best_x
        rng = np.random.default_rng(12345)
        for attempt in range(5):
            if call_count > 440 or not time_ok():
                break

            # Try different perturbation scales
            scale = 0.5 * (attempt + 1)
            x_try = best_x + rng.normal(0, scale, size=2)

            try:
                Fx_try = F_count(x_try)
            except:
                continue

            if not np.all(np.isfinite(Fx_try)):
                continue

            nr = float(np.dot(Fx_try, Fx_try))
            if nr < best_norm_sq:
                best_x = x_try.copy()
                best_norm_sq = nr
                if nr <= tol_sq:
                    return best_x

                # Run Broyden from this new point
                x = x_try.copy()
                Fx = Fx_try
                norm_sq = nr
                J = compute_jacobian_fd(x, Fx)
                if J is None:
                    J = np.eye(2)
                lambda_lm = 1e-3
                stagnation_count = 0

                for _ in range(60):
                    if call_count > 480 or not time_ok():
                        break
                    if norm_sq <= tol_sq:
                        return x

                    try:
                        JtJ = J.T @ J
                        JtF = J.T @ Fx
                        A = JtJ + lambda_lm * np.eye(2)
                        s = np.linalg.solve(A, -JtF)
                    except np.linalg.LinAlgError:
                        lambda_lm *= 10.0
                        continue

                    s_norm = np.linalg.norm(s)
                    if s_norm > 5.0:
                        s = s * (5.0 / s_norm)

                    x_try2 = x + s
                    try:
                        Fx_try2 = F_count(x_try2)
                    except:
                        lambda_lm *= 10.0
                        continue

                    if not np.all(np.isfinite(Fx_try2)):
                        lambda_lm *= 10.0
                        continue

                    nr2 = float(np.dot(Fx_try2, Fx_try2))
                    if nr2 < norm_sq:
                        dx = x_try2 - x
                        dF = Fx_try2 - Fx
                        Jdx = J @ dx
                        residual = dF - Jdx
                        dx_norm_sq = float(np.dot(dx, dx))
                        if dx_norm_sq > 1e-30:
                            J = J + np.outer(residual, dx) / dx_norm_sq
                        x = x_try2
                        Fx = Fx_try2
                        norm_sq = nr2
                        lambda_lm = max(1e-12, lambda_lm * 0.3)
                        if norm_sq < best_norm_sq:
                            best_norm_sq = norm_sq
                            best_x = x.copy()
                    else:
                        lambda_lm *= 10.0

    # Final check: if we have a good enough result, return it
    if best_norm_sq <= tol_sq:
        return best_x

    # Try one more thing: if best is close, do a few more Broyden steps from best_x
    if best_norm_sq < 1e-4 and call_count < 490 and time_ok():
        x = best_x.copy()
        try:
            Fx = F_count(x)
        except:
            return best_x
        if not np.all(np.isfinite(Fx)):
            return best_x
        norm_sq = float(np.dot(Fx, Fx))
        J = compute_jacobian_fd(x, Fx)
        if J is None:
            J = np.eye(2)
        lambda_lm = 1e-4

        for _ in range(30):
            if call_count > 495 or not time_ok():
                break
            if norm_sq <= tol_sq:
                return x

            try:
                JtJ = J.T @ J
                JtF = J.T @ Fx
                A = JtJ + lambda_lm * np.eye(2)
                s = np.linalg.solve(A, -JtF)
            except:
                lambda_lm *= 10.0
                continue

            s_norm = np.linalg.norm(s)
            if s_norm > 1.0:
                s = s * (1.0 / s_norm)

            x_try = x + s
            try:
                Fx_try = F_count(x_try)
            except:
                lambda_lm *= 10.0
                continue

            if not np.all(np.isfinite(Fx_try)):
                lambda_lm *= 10.0
                continue

            nr = float(np.dot(Fx_try, Fx_try))
            if nr < norm_sq:
                dx = x_try - x
                dF = Fx_try - Fx
                Jdx = J @ dx
                residual = dF - Jdx
                dx_norm_sq = float(np.dot(dx, dx))
                if dx_norm_sq > 1e-30:
                    J = J + np.outer(residual, dx) / dx_norm_sq
                x = x_try
                Fx = Fx_try
                norm_sq = nr
                lambda_lm = max(1e-14, lambda_lm * 0.3)
                if norm_sq < best_norm_sq:
                    best_norm_sq = norm_sq
                    best_x = x.copy()
            else:
                lambda_lm *= 10.0

    return best_x

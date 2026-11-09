"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    def norm(vec):
        return np.linalg.norm(vec)

    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False

    def finite_diff_jac(F, x, Fx, eps=1e-6):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            try:
                J[:, j] = (F(x + e) - Fx) / eps
            except Exception:
                return None
        if not np.all(np.isfinite(J)):
            return None
        return J

    def newton_damped(F, x0, max_iter=80, tol=1e-12):
        x = np.array(x0, dtype=float)
        best_x = x.copy()
        best_norm = np.inf

        for i in range(max_iter):
            if time_left() < 0.3:
                break

            try:
                Fx = F(x)
            except Exception:
                break

            if not np.all(np.isfinite(Fx)):
                break

            current_norm = norm(Fx)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x.copy()

            if current_norm <= tol:
                return x

            # Compute Jacobian
            J = finite_diff_jac(F, x, Fx)
            if J is None:
                break

            # Check conditioning
            cond = np.linalg.cond(J)
            if not np.isfinite(cond) or cond > 1e12:
                # Try with larger eps
                J = finite_diff_jac(F, x, Fx, eps=1e-5)
                if J is None:
                    break
                cond = np.linalg.cond(J)
                if not np.isfinite(cond) or cond > 1e14:
                    break

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break

            if not np.all(np.isfinite(dx)):
                break

            # Damped Newton Step (Line Search)
            step_found = False
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]:
                x_new = x + lam * dx
                try:
                    F_new = F(x_new)
                except Exception:
                    continue

                if not np.all(np.isfinite(F_new)):
                    continue

                new_norm = norm(F_new)
                if new_norm < current_norm * (1 - 1e-4 * lam * norm(dx)) or new_norm < current_norm:
                    x = x_new
                    step_found = True
                    break

            if not step_found:
                break

            if norm(dx) < 1e-12:
                break

        return best_x

    def lm_step(F, x, Fx, J, g, max_ls=10):
        """Levenberg-Marquardt style step."""
        # Solve (J^T J + lambda * I) dx = -J^T Fx
        JtJ = J.T @ J
        JtF = J.T @ Fx
        n = 2
        I = np.eye(n)

        # Initial lambda based on diagonal of JtJ
        diag = np.diag(JtJ)
        if np.all(diag > 1e-10):
            lam = 0.1 * np.mean(diag)
        else:
            lam = 1e-4

        current_norm = norm(Fx)
        best_dx = None
        best_new_norm = np.inf

        for _ in range(max_ls):
            A = JtJ + lam * I
            try:
                dx = np.linalg.solve(A, -JtF)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(A, -JtF, rcond=None)[0]
                except Exception:
                    break

            if not np.all(np.isfinite(dx)):
                lam *= 10
                continue

            x_new = x + dx
            try:
                F_new = F(x_new)
            except Exception:
                lam *= 10
                continue

            if not np.all(np.isfinite(F_new)):
                lam *= 10
                continue

            new_norm = norm(F_new)
            if new_norm < best_new_norm:
                best_new_norm = new_norm
                best_dx = dx

            if new_norm < current_norm:
                return dx, x_new, new_norm

            lam *= 10
            if lam > 1e12:
                break
        else:
            if best_dx is not None:
                return best_dx, x + best_dx, best_new_norm
            return None, x, np.inf

        return best_dx, x + best_dx if best_dx is not None else x, best_new_norm

    def newton_hybrid(F, x0, max_iter=80, tol=1e-12):
        """Newton with LM fallback when Newton step fails."""
        x = np.array(x0, dtype=float)
        best_x = x.copy()
        best_norm = np.inf
        use_lm = False

        for i in range(max_iter):
            if time_left() < 0.3:
                break

            try:
                Fx = F(x)
            except Exception:
                break

            if not np.all(np.isfinite(Fx)):
                break

            current_norm = norm(Fx)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x.copy()

            if current_norm <= tol:
                return x

            J = finite_diff_jac(F, x, Fx)
            if J is None:
                if use_lm:
                    break
                use_lm = True
                continue

            if not use_lm:
                # Try plain Newton first
                cond = np.linalg.cond(J)
                if np.isfinite(cond) and cond < 1e10:
                    try:
                        dx = np.linalg.solve(J, -Fx)
                    except np.linalg.LinAlgError:
                        dx = None

                    if dx is not None and np.all(np.isfinite(dx)):
                        # Line search
                        step_found = False
                        for lam in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                            x_new = x + lam * dx
                            try:
                                F_new = F(x_new)
                            except Exception:
                                continue
                            if not np.all(np.isfinite(F_new)):
                                continue
                            new_norm = norm(F_new)
                            if new_norm < current_norm:
                                x = x_new
                                step_found = True
                                if norm(dx) < 1e-12:
                                    break
                                break

                        if step_found:
                            continue

            # LM fallback
            dx, x_new, new_norm = lm_step(F, x, Fx, J, Fx)
            if dx is not None:
                x = x_new
                if norm(dx) < 1e-12:
                    break
            else:
                break

        return best_x

    # Phase 1: Try Newton from x0
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])

    x_best = newton_hybrid(F, x0)

    if is_good(x_best):
        return x_best

    # Phase 2: Random Restarts
    scale = max(1.0, norm(x0))

    candidates = []

    # Zero
    candidates.append(np.array([0.0, 0.0]))

    # x0 scaled
    candidates.append(x0 * 0.5)
    candidates.append(x0 * 2.0)
    candidates.append(x0 * 0.1)

    # Random points
    rng = np.random.default_rng(42)

    for r in [scale * 0.01, scale * 0.05, scale * 0.1, scale * 0.25, scale * 0.5, scale * 1.0, scale * 2.0, scale * 4.0]:
        for _ in range(4):
            u = rng.normal(size=2)
            u = u / norm(u)
            p = x0 + r * u
            candidates.append(p)

    # Also try some random points far away
    for r in [scale * 5.0, scale * 10.0, scale * 20.0]:
        for _ in range(3):
            u = rng.normal(size=2)
            u = u / norm(u)
            p = r * u
            candidates.append(p)

    # Sort by distance to x0 for locality
    candidates.sort(key=lambda c: norm(c - x0))

    # Keep track of best
    try:
        F_best = F(x_best)
        best_global_norm = norm(F_best) if np.all(np.isfinite(F_best)) else np.inf
    except Exception:
        best_global_norm = np.inf

    for c in candidates:
        if time_left() < 1.0:
            break

        if not np.all(np.isfinite(c)):
            continue

        x_try = newton_hybrid(F, c)

        if is_good(x_try):
            return x_try

        try:
            F_try = F(x_try)
            if np.all(np.isfinite(F_try)):
                n_try = norm(F_try)
                if n_try < best_global_norm:
                    best_global_norm = n_try
                    x_best = x_try
        except Exception:
            pass

    return x_best
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def is_good(x):
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return False
            Fx = F(x)
            return np.linalg.norm(Fx) <= 1e-8
        except Exception:
            return False

    def broyden_damped(x, max_iter=80):
        """Broyden's method with damped line search."""
        x = np.array(x, dtype=float)
        Fx = F(x)
        fnorm = np.linalg.norm(Fx)
        if fnorm <= 1e-8:
            return x, True

        # Initial Jacobian via finite differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-6
            J[:, j] = (F(x + e) - Fx) / 1e-6

        best_x = x
        best_fnorm = fnorm

        for it in range(max_iter):
            if time_left() < 2.0:
                break

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break

            # Damped line search
            alpha = 1.0
            found = False
            for _ in range(20):
                x_new = x + alpha * dx
                Fx_new = F(x_new)
                fnorm_new = np.linalg.norm(Fx_new)

                if not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(Fx_new)):
                    alpha *= 0.5
                    continue

                # Armijo-like condition: sufficient decrease
                if fnorm_new < fnorm * (1 - 1e-4 * alpha):
                    found = True
                    break

                if fnorm_new < best_fnorm:
                    best_x = x_new.copy()
                    best_fnorm = fnorm_new

                alpha *= 0.5
                if alpha < 1e-10:
                    break

            if not found:
                # If no good step found, try a smaller step or break
                if alpha < 1e-10:
                    break
                x_new = x + alpha * dx
                Fx_new = F(x_new)
                fnorm_new = np.linalg.norm(Fx_new)
                if not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(Fx_new)):
                    break
                if fnorm_new >= fnorm:
                    break

            # Update Jacobian (Broyden rank-1 update)
            s = x_new - x
            y = Fx_new - Fx
            if np.linalg.norm(s) > 1e-15:
                J = J + np.outer(y - J @ s, s) / (s @ s)

            x = x_new
            Fx = Fx_new
            fnorm = fnorm_new

            if fnorm < best_fnorm:
                best_x = x.copy()
                best_fnorm = fnorm_new

            if fnorm <= 1e-8:
                return x, True

        return best_x, best_fnorm <= 1e-8

    def global_search(x0, max_iter=50):
        """Simple global search via random perturbations + local refinement."""
        x_best = np.array(x0, dtype=float)
        Fx_best = F(x_best)
        norm_best = np.linalg.norm(Fx_best)

        rng = np.random.RandomState(42)
        for it in range(max_iter):
            if time_left() < 2.0:
                break

            # Random perturbation around x0
            scale = 1.0 * (0.5 ** (it // 10))  # decreasing scale
            x_try = x0 + rng.normal(0, scale, size=2)

            Fx_try = F(x_try)
            norm_try = np.linalg.norm(Fx_try)

            if norm_try < norm_best:
                x_best = x_try
                norm_best = norm_try

            # If close enough, refine with Broyden
            if norm_try < 1.0:
                x_refined, success = broyden_damped(x_try, max_iter=40)
                Fx_ref = F(x_refined)
                norm_ref = np.linalg.norm(Fx_ref)
                if norm_ref < norm_best:
                    x_best = x_refined
                    norm_best = norm_ref
                    if norm_ref <= 1e-8:
                        return x_best

        return x_best

    # Try Broyden's method first from x0
    x_result, success = broyden_damped(x0, max_iter=80)

    if success:
        return x_result

    # If Broyden didn't succeed, try global search
    if time_left() > 5.0:
        x_global = global_search(x0, max_iter=50)
        Fx_global = F(x_global)
        if np.linalg.norm(Fx_global) < np.linalg.norm(F(x_result)):
            x_result = x_global
            success = np.linalg.norm(Fx_global) <= 1e-8

    # Final polish: if close, do a high-precision Newton
    if not success:
        Fx_final = F(x_result)
        if np.linalg.norm(Fx_final) < 1e-4 and time_left() > 2.0:
            # High-precision Newton with tighter tolerance
            x_polish = np.array(x_result, dtype=float)
            for _ in range(30):
                if time_left() < 1.0:
                    break
                Fx_p = F(x_polish)
                if np.linalg.norm(Fx_p) <= 1e-8:
                    x_result = x_polish
                    success = True
                    break
                h = 1e-8
                J_p = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = 1e-7
                    J_p[:, j] = (F(x_polish + e) - Fx_p) / 1e-7
                try:
                    dx_p = np.linalg.solve(J_p, -Fx_p)
                    x_polish = x_polish + dx_p
                except np.linalg.LinAlgError:
                    break

    return x_result
# EVOLVE-BLOCK-END

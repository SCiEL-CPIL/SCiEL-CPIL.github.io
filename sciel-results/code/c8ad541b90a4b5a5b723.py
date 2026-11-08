"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _start_time) < _DEADLINE


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    # Try Newton's method with adaptive step and line search
    for _ in range(100):
        if not _time_ok():
            break

        try:
            Fx = F(x)
        except Exception:
            break

        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            break

        fx_norm = np.linalg.norm(Fx)
        if fx_norm < best_norm:
            best_norm = fx_norm
            best_x = x.copy()

        if fx_norm <= 1e-10:
            return x

        # Compute Jacobian via central differences for better accuracy
        h = 1e-5
        J = np.zeros((2, 2))
        try:
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fp = F(x + e)
                Fm = F(x - e)
                Fp = np.asarray(Fp, dtype=float)
                Fm = np.asarray(Fm, dtype=float)
                J[:, j] = (Fp - Fm) / (2 * h)
        except Exception:
            break

        if not np.all(np.isfinite(J)):
            break

        # Solve for Newton step
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(step)):
            break

        # Line search with backtracking
        alpha = 1.0
        converged = False
        for _ in range(20):
            x_new = x - alpha * step
            try:
                Fx_new = F(x_new)
                Fx_new = np.asarray(Fx_new, dtype=float)
                if Fx_new.shape != (2,) or not np.all(np.isfinite(Fx_new)):
                    alpha *= 0.5
                    continue
                fx_new_norm = np.linalg.norm(Fx_new)
            except Exception:
                alpha *= 0.5
                continue

            if fx_new_norm < fx_norm:
                x = x_new
                if fx_new_norm < best_norm:
                    best_norm = fx_new_norm
                    best_x = x.copy()
                if fx_new_norm <= 1e-10:
                    return x
                converged = True
                break
            alpha *= 0.5

        if not converged:
            # If line search fails, try damped Newton
            x = x - 0.1 * step
            if not np.all(np.isfinite(x)):
                break

    return best_x
# EVOLVE-BLOCK-END

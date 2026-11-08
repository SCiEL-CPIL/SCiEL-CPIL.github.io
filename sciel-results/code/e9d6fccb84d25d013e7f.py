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

    # Initial evaluation
    try:
        Fx = F(x)
    except Exception:
        return best_x
    Fx = np.asarray(Fx, dtype=float)
    if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
        return best_x
    
    fx_norm = np.linalg.norm(Fx)
    if fx_norm < best_norm:
        best_norm = fx_norm
        best_x = x.copy()
        
    if fx_norm <= 1e-10:
        return x

    # Initialize Jacobian with Forward Differences (2 extra calls)
    h = 1e-6
    J = np.zeros((2, 2))
    try:
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fp = F(x + e)
            Fp = np.asarray(Fp, dtype=float)
            if Fp.shape != (2,) or not np.all(np.isfinite(Fp)):
                raise ValueError("Non-finite F")
            J[:, j] = (Fp - Fx) / h
    except Exception:
        # Fallback to simple Newton if Jacobian init fails
        return _fallback_newton(F, x, Fx, best_x, best_norm)

    if not np.all(np.isfinite(J)):
        return _fallback_newton(F, x, Fx, best_x, best_norm)

    # Broyden's Good Method
    max_iters = 50
    for _ in range(max_iters):
        if not _time_ok():
            break

        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, refresh J with FD and try Newton step
            return _fallback_newton(F, x, Fx, best_x, best_norm)

        if not np.all(np.isfinite(s)):
            break

        # Line search
        alpha = 1.0
        converged = False
        for _ in range(20):
            x_new = x + alpha * s
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
                # Update Broyden's Jacobian
                # y = F_new - F_old
                # s_actual = x_new - x
                y = Fx_new - Fx
                s_actual = x_new - x
                denom = s_actual @ s_actual
                if denom > 1e-15:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom
                
                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm
                
                if fx_norm < best_norm:
                    best_norm = fx_norm
                    best_x = x.copy()
                
                if fx_norm <= 1e-10:
                    return x
                
                converged = True
                break
            alpha *= 0.5

        if not converged:
            # If line search fails, try a smaller damped Newton step or break
            # Let's try a small step in the direction of -Fx (gradient descent on norm)
            # Or just break and return best
            break

    return best_x

def _fallback_newton(F, x, Fx, best_x, best_norm):
    """Fallback to standard Newton with Forward Differences if Broyden fails."""
    for _ in range(20):
        if not _time_ok():
            break
        try:
            # Recompute Jacobian
            h = 1e-5
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fp = F(x + e)
                Fp = np.asarray(Fp, dtype=float)
                if Fp.shape != (2,) or not np.all(np.isfinite(Fp)):
                    return best_x
                J[:, j] = (Fp - Fx) / h
            
            s = np.linalg.solve(J, -Fx)
        except Exception:
            return best_x
        
        if not np.all(np.isfinite(s)):
            return best_x
            
        alpha = 1.0
        converged = False
        for _ in range(20):
            x_new = x + alpha * s
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
                
            if fx_new_norm < np.linalg.norm(Fx):
                x = x_new
                Fx = Fx_new
                if fx_new_norm < best_norm:
                    best_norm = fx_new_norm
                    best_x = x.copy()
                if fx_new_norm <= 1e-10:
                    return x
                converged = True
                break
            alpha *= 0.5
        if not converged:
            break
            
    return best_x
# EVOLVE-BLOCK-END

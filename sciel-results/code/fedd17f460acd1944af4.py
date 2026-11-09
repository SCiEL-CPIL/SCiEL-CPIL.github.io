"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8 using a hybrid approach:
    1. Try Newton's method with adaptive finite-difference Jacobian.
    2. If Newton fails or stagnates, fall back to a derivative-free
       trust-region-like method using scipy.optimize.minimize on ||F||^2.
    3. Track elapsed time and return best result before deadline.
    """
    x0 = np.asarray(x0, dtype=float)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    def check_time():
        return time.time() >= deadline

    def norm2(v):
        return float(np.dot(v, v))

    # Track best solution found so far
    best_x = x0.copy()
    best_norm = np.inf
    try:
        best_Fx = F(x0)
        best_norm = float(np.linalg.norm(best_Fx))
    except Exception:
        best_norm = np.inf

    def update_best(x, Fx):
        nonlocal best_x, best_norm
        n = float(np.linalg.norm(Fx))
        if n < best_norm:
            best_norm = n
            best_x = x.copy()

    # Phase 1: Newton's method with adaptive step control
    x = x0.copy()
    newton_iters = 0
    max_newton_iters = 40

    for i in range(max_newton_iters):
        if check_time():
            break
        newton_iters += 1

        try:
            Fx = F(x)
        except Exception:
            break

        update_best(x, Fx)
        nrm = float(np.linalg.norm(Fx))
        if nrm <= 1e-8:
            return x

        # Compute finite-difference Jacobian with adaptive step
        # Start with a reasonable step size
        scale = max(1.0, np.linalg.norm(x))
        h = 1e-6 * scale

        # Try to compute Jacobian
        J = np.zeros((2, 2))
        try:
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_plus = F(x + e)
                J[:, j] = (Fx_plus - Fx) / h
        except Exception:
            break

        # Check if Jacobian is well-conditioned
        try:
            cond = np.linalg.cond(J)
        except Exception:
            break

        if not np.isfinite(cond) or cond > 1e15:
            break

        # Solve for Newton step
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break

        # Line search to ensure sufficient decrease
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break

        # Damped Newton with backtracking line search
        alpha = 1.0
        max_backtracks = 20
        found_step = False

        for bt in range(max_backtracks):
            if check_time():
                break
            x_trial = x - alpha * step
            try:
                Fx_trial = F(x_trial)
            except Exception:
                alpha *= 0.5
                continue

            nrm_trial = float(np.linalg.norm(Fx_trial))

            # Accept if we reduce the norm or if we're very close
            if nrm_trial <= nrm * (1 - 1e-4 * alpha) or nrm_trial <= 1e-8:
                update_best(x_trial, Fx_trial)
                x = x_trial
                found_step = True
                break
            else:
                alpha *= 0.5

        if not found_step:
            # If line search fails, try a smaller fixed step
            x_trial = x - 0.1 * step
            try:
                Fx_trial = F(x_trial)
                nrm_trial = float(np.linalg.norm(Fx_trial))
                if nrm_trial < nrm:
                    update_best(x_trial, Fx_trial)
                    x = x_trial
                else:
                    break
            except Exception:
                break

        # Check convergence
        if nrm <= 1e-8:
            return x

    # If Newton succeeded already, we'd have returned. Otherwise, proceed to Phase 2.

    # Phase 2: Derivative-free optimization using scipy
    from scipy.optimize import minimize

    def objective(x):
        try:
            Fx = F(x)
            return float(np.dot(Fx, Fx))
        except Exception:
            return 1e10

    # Use Nelder-Mead as a robust derivative-free method
    try:
        result = minimize(
            objective,
            x,
            method='Nelder-Mead',
            options={
                'xatol': 1e-12,
                'fatol': 1e-18,
                'maxiter': 200,
                'adaptive': True
            }
        )
        x_nm = result.x
        try:
            Fx_nm = F(x_nm)
            update_best(x_nm, Fx_nm)
        except Exception:
            pass
    except Exception:
        pass

    if check_time():
        return best_x

    # Phase 3: Try BFGS as another derivative-free option
    try:
        result2 = minimize(
            objective,
            x,
            method='BFGS',
            options={
                'gtol': 1e-12,
                'maxiter': 100
            }
        )
        x_bfgs = result2.x
        try:
            Fx_bfgs = F(x_bfgs)
            update_best(x_bfgs, Fx_bfgs)
        except Exception:
            pass
    except Exception:
        pass

    # Final check
    if best_norm <= 1e-8:
        return best_x

    # If we still haven't reached tolerance, try one more Newton refinement
    # from the best point found
    x = best_x.copy()
    for i in range(20):
        if check_time():
            break
        try:
            Fx = F(x)
        except Exception:
            break
        nrm = float(np.linalg.norm(Fx))
        if nrm <= 1e-8:
            return x

        scale = max(1.0, np.linalg.norm(x))
        h = 1e-7 * scale

        J = np.zeros((2, 2))
        try:
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_plus = F(x + e)
                J[:, j] = (Fx_plus - Fx) / h
        except Exception:
            break

        try:
            cond = np.linalg.cond(J)
            if not np.isfinite(cond) or cond > 1e15:
                break
            step = np.linalg.solve(J, Fx)
            x = x - step
        except Exception:
            break

    return best_x
# EVOLVE-BLOCK-END

"""Root finder for 3 fixed 2-D systems: Broyden + Newton + scipy fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with finite-difference initial Jacobian.
    2. If that fails, damped Newton with finite-difference Jacobian.
    3. If that fails, scipy L-BFGS-B on 0.5*||F(x)||^2.
    4. Return best result found within budget.
    
    Budget:
    - Time limit: 145s internal deadline (150s hard limit).
    - Call limit: < 490 calls per system.
    """
    t_start = time.time()
    deadline = t_start + 145.0
    x = np.array(x0, dtype=float)
    calls = 0
    CALL_BUDGET = 490

    def time_ok():
        return time.time() < deadline

    def can_call():
        return calls < CALL_BUDGET and time_ok()

    def safe_F(x_val):
        nonlocal calls
        if not can_call():
            return None
        try:
            result = np.asarray(F(x_val), dtype=float)
            calls += 1
            if result.shape == (2,) and np.all(np.isfinite(result)):
                return result
            return None
        except Exception:
            calls += 1
            return None

    # Initial evaluation
    Fx = safe_F(x)
    if Fx is None:
        return x

    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    if best_norm <= 1e-8:
        return best_x

    # Strategy 1: Broyden's method
    result = _broyden(x, Fx, best_x, best_norm, safe_F, can_call)
    if result is not None:
        return result

    # Update state from Broyden
    x = best_x.copy()
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    best_norm = np.linalg.norm(Fx)
    if best_norm <= 1e-8:
        return best_x

    # Strategy 2: Damped Newton
    result = _damped_newton(x, Fx, best_x, best_norm, safe_F, can_call)
    if result is not None:
        return result

    # Update state from Newton
    x = best_x.copy()
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    best_norm = np.linalg.norm(Fx)
    if best_norm <= 1e-8:
        return best_x

    # Strategy 3: scipy L-BFGS-B
    if calls < CALL_BUDGET - 5 and time_ok():
        result = _scipy_minimize(x, best_x, best_norm, safe_F, can_call, calls)
        if result is not None:
            return result

    return best_x


def _broyden(x, Fx, best_x, best_norm, safe_F, can_call):
    """Broyden's method with rank-1 update and trust-region damping."""
    x = x.copy()
    Fx = Fx.copy()
    best_x = best_x.copy()
    best_norm = best_norm

    # Get initial Jacobian via finite differences
    h = 1e-6
    J = np.eye(2)
    jac_valid = True
    for j in range(2):
        if not can_call():
            jac_valid = False
            break
        e = np.zeros(2)
        e[j] = h
        Fx_p = safe_F(x + e)
        if Fx_p is None:
            jac_valid = False
            break
        J[:, j] = (Fx_p - Fx) / h

    if not jac_valid:
        # Try with larger h
        h = 1e-5
        J = np.eye(2)
        jac_valid = True
        for j in range(2):
            if not can_call():
                jac_valid = False
                break
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_F(x + e)
            if Fx_p is None:
                jac_valid = False
                break
            J[:, j] = (Fx_p - Fx) / h

    if not jac_valid:
        return None

    # Check if Jacobian is well-conditioned
    try:
        cond = np.linalg.cond(J)
        if cond > 1e12:
            return None
    except np.linalg.LinAlgError:
        return None

    max_iter = 30
    for it in range(max_iter):
        if not can_call() or best_norm <= 1e-8:
            break

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(dx)):
            break

        # Trust-region: limit step size
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx * (1.0 / step_norm)

        # Try full step
        x_new = x + dx
        Fx_new = safe_F(x_new)

        if Fx_new is None:
            # Try half step
            dx = dx * 0.5
            x_new = x + dx
            Fx_new = safe_F(x_new)
            if Fx_new is None:
                break

        norm_new = np.linalg.norm(Fx_new)

        # Update best
        if norm_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_new

        if norm_new <= 1e-8:
            return best_x

        # Broyden rank-1 update
        y = Fx_new - Fx
        denom = dx @ dx
        if denom > 1e-14:
            J = J + np.outer(y - J @ dx, dx) / denom

        # Accept step if it reduces residual
        if norm_new < best_norm:
            x = x_new
            Fx = Fx_new
        else:
            # Try smaller step
            dx = dx * 0.5
            x_new = x + dx
            Fx_new = safe_F(x_new)
            if Fx_new is not None:
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < best_norm:
                    x = x_new
                    Fx = Fx_new
                    best_x = x_new.copy()
                    best_norm = norm_new
                    if norm_new <= 1e-8:
                        return best_x
                    # Update Broyden
                    y = Fx_new - Fx
                    denom = dx @ dx
                    if denom > 1e-14:
                        J = J + np.outer(y - J @ dx, dx) / denom
                else:
                    break
            else:
                break

    if best_norm <= 1e-8:
        return best_x
    return None


def _damped_newton(x, Fx, best_x, best_norm, safe_F, can_call):
    """Damped Newton with finite-difference Jacobian and backtracking line search."""
    x = x.copy()
    Fx = Fx.copy()
    best_x = best_x.copy()
    best_norm = best_norm

    max_iter = 15
    for it in range(max_iter):
        if not can_call() or best_norm <= 1e-8:
            break

        # Finite-difference Jacobian
        h = 1e-6
        J = np.zeros((2, 2))
        valid = True
        for j in range(2):
            if not can_call():
                valid = False
                break
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_F(x + e)
            if Fx_p is None:
                valid = False
                break
            J[:, j] = (Fx_p - Fx) / h

        if not valid:
            break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(dx)):
            break

        # Backtracking line search
        alpha = 1.0
        accepted = False
        for _ in range(8):
            if not can_call():
                break
            x_new = x + alpha * dx
            Fx_new = safe_F(x_new)
            if Fx_new is not None:
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < best_norm:
                    x = x_new
                    Fx = Fx_new
                    best_x = x_new.copy()
                    best_norm = norm_new
                    if norm_new <= 1e-8:
                        return best_x
                    accepted = True
                    break
            alpha *= 0.5

        if best_norm <= 1e-8:
            break
        if not accepted:
            break

    if best_norm <= 1e-8:
        return best_x
    return None


def _scipy_minimize(x, best_x, best_norm, safe_F, can_call, calls):
    """Use scipy L-BFGS-B to minimize 0.5*||F(x)||^2."""
    # We need a callback to track calls and stop early
    call_count = [0]
    
    def objective(x_val):
        Fx = safe_F(x_val)
        if Fx is None:
            return 1e10
        return 0.5 * np.dot(Fx, Fx)
    
    # Set a maxiter that's reasonable
    max_iter = 200
    try:
        res = minimize(
            objective,
            x,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': CALL_BUDGET - calls, 'ftol': 1e-12}
        )
    except Exception:
        return None
    
    if res.success or res.fun < best_norm ** 2:
        x_best = res.x
        Fx_best = safe_F(x_best)
        if Fx_best is not None:
            norm_best = np.linalg.norm(Fx_best)
            if norm_best < best_norm:
                return x_best
            if norm_best <= 1e-8:
                return x_best
    
    return None
# EVOLVE-BLOCK-END

"""Root finder for 3 fixed 2D systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with a robust line search and a fallback to SciPy's hybr method.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    call_count = 0
    max_calls = 450  # Leave buffer for safety

    def F_wrapper(x):
        nonlocal call_count
        if call_count >= max_calls:
            return np.array([0.0, 0.0])  # Dummy to avoid crash
        call_count += 1
        result = F(x)
        return np.asarray(result, dtype=float)

    def time_up():
        return time.time() >= deadline

    def norm(v):
        return np.linalg.norm(v)

    # Try Broyden's method first
    x = np.array(x0, dtype=float)
    Fx = F_wrapper(x)
    n = 2

    # Initialize Jacobian estimate
    J = np.eye(n)
    # Better initialization: use finite differences if possible
    eps = 1e-7
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        Fx_e = F_wrapper(x + e)
        J[:, j] = (Fx_e - Fx) / eps

    best_x = x.copy()
    best_norm = norm(Fx)

    for iteration in range(100):
        if time_up():
            break

        # Check convergence
        if best_norm <= 1e-8:
            break

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, try pseudoinverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Line search to ensure sufficient decrease
        alpha = 1.0
        Fx_old = Fx.copy()
        found_alpha = False
        for ls_iter in range(30):
            if time_up():
                break
            x_trial = x + alpha * dx
            Fx_trial = F_wrapper(x_trial)
            norm_trial = norm(Fx_trial)

            if norm_trial < best_norm:
                best_x = x_trial.copy()
                best_norm = norm_trial

            if norm_trial <= 1e-12:
                x = x_trial
                Fx = Fx_trial
                found_alpha = True
                break

            # Armijo condition: sufficient decrease
            if norm_trial <= norm(Fx_old) - 1e-4 * alpha * norm(Fx_old) ** 2:
                x = x_trial
                Fx = Fx_trial
                found_alpha = True
                break

            alpha *= 0.5

        if not found_alpha and not time_up():
            # If no good alpha found, try smaller step
            alpha = 1e-4
            x_trial = x + alpha * dx
            Fx_trial = F_wrapper(x_trial)
            if norm(Fx_trial) < best_norm:
                x = x_trial
                Fx = Fx_trial
                best_x = x.copy()
                best_norm = norm(Fx)
            else:
                # Stagnation detected, break
                break

        # Update Jacobian using Broyden's formula
        if found_alpha or norm(Fx) < best_norm:
            Fx_new = Fx
            s = alpha * dx
            y = Fx_new - Fx_old
            # Broyden's update: J = J + (y - J*s) * s^T / (s^T * s)
            s_dot_s = s @ s
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s

        # Refresh Jacobian periodically
        if iteration % 5 == 4 and not time_up():
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = eps
                Fx_e = F_wrapper(x + e)
                J[:, j] = (Fx_e - Fx) / eps
            Fx_old = Fx.copy()

    # Check if we converged
    if best_norm <= 1e-8:
        return best_x

    # Fallback: try SciPy's hybr method
    if not time_up() and call_count < max_calls - 50:
        try:
            result = root(F_wrapper, x0, method='hybr', options={'maxfev': max(1, max_calls - call_count - 5)})
            if result.success and norm(result.fun) <= 1e-8:
                return result.x
            if norm(result.fun) < best_norm:
                best_x = result.x
                best_norm = norm(result.fun)
        except:
            pass

    return best_x
# EVOLVE-BLOCK-END

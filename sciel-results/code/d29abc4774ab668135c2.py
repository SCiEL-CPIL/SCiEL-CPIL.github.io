"""Root finder for 3 fixed 2D systems using Broyden's method with monotonic line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with finite difference initialization and monotonic line search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    x = np.array(x0, dtype=float)
    if x.shape != (2,):
        x = x.reshape(2)

    # Initial function evaluation
    Fx = np.array(F(x), dtype=float)
    n_calls = 1
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    if best_norm <= 1e-8:
        return x

    # Constants
    max_iter = 150
    fd_h = 1e-5
    max_ls_steps = 8
    step_limit = 10.0
    polish_tol = 1e-4

    def compute_fd_jacobian(x_curr, F_curr):
        J_fd = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = fd_h
            Fx_e = np.array(F(x_curr + e), dtype=float)
            J_fd[:, j] = (Fx_e - F_curr) / fd_h
        return J_fd

    # Compute initial Jacobian
    J = compute_fd_jacobian(x, Fx)
    n_calls += 2

    # Broyden's Method Loop
    for iteration in range(max_iter):
        # Check deadline
        if time.time() > deadline:
            break

        # Check convergence
        if best_norm <= 1e-8:
            return best_x

        # If we are close, use a more aggressive polish strategy
        if best_norm < polish_tol:
            # Try full Newton step with current J
            try:
                s = -np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                J = compute_fd_jacobian(x, Fx)
                n_calls += 2
                s = -np.linalg.solve(J, Fx)
            
            # Limit step size
            step_norm = np.linalg.norm(s)
            if step_norm > step_limit:
                s = s / step_norm * step_limit

            # Try the full step
            x_trial = x + s
            F_trial = np.array(F(x_trial), dtype=float)
            n_calls += 1
            
            if not np.all(np.isfinite(F_trial)):
                # Backtrack
                alpha = 0.5
                for _ in range(5):
                    x_trial = x + alpha * s
                    F_trial = np.array(F(x_trial), dtype=float)
                    n_calls += 1
                    if np.all(np.isfinite(F_trial)):
                        norm_trial = np.linalg.norm(F_trial)
                        if norm_trial < best_norm:
                            best_x = x_trial.copy()
                            best_norm = norm_trial
                        if norm_trial < np.linalg.norm(Fx):
                            # Accept and update
                            s_actual = x_trial - x
                            y = F_trial - Fx
                            if s_actual @ s_actual > 1e-15:
                                J = J + np.outer(y - J @ s_actual, s_actual) / (s_actual @ s_actual)
                            x = x_trial
                            Fx = F_trial
                            break
                        alpha *= 0.5
                continue
            else:
                norm_trial = np.linalg.norm(F_trial)
                if norm_trial <= 1e-8:
                    return x_trial
                if norm_trial < best_norm:
                    best_x = x_trial.copy()
                    best_norm = norm_trial
                if norm_trial < np.linalg.norm(Fx):
                    # Accept and update
                    s_actual = x_trial - x
                    y = F_trial - Fx
                    if s_actual @ s_actual > 1e-15:
                        J = J + np.outer(y - J @ s_actual, s_actual) / (s_actual @ s_actual)
                    x = x_trial
                    Fx = F_trial
                    continue
                else:
                    # Backtrack
                    alpha = 0.5
                    for _ in range(5):
                        x_trial = x + alpha * s
                        F_trial = np.array(F(x_trial), dtype=float)
                        n_calls += 1
                        if not np.all(np.isfinite(F_trial)):
                            alpha *= 0.5
                            continue
                        norm_trial = np.linalg.norm(F_trial)
                        if norm_trial < best_norm:
                            best_x = x_trial.copy()
                            best_norm = norm_trial
                        if norm_trial < np.linalg.norm(Fx):
                            # Accept and update
                            s_actual = x_trial - x
                            y = F_trial - Fx
                            if s_actual @ s_actual > 1e-15:
                                J = J + np.outer(y - J @ s_actual, s_actual) / (s_actual @ s_actual)
                            x = x_trial
                            Fx = F_trial
                            break
                        alpha *= 0.5
                    continue

        # Normal Broyden step
        try:
            # Check condition number roughly; if ill-conditioned, recompute J
            cond = np.linalg.cond(J)
            if cond > 1e12:
                J = compute_fd_jacobian(x, Fx)
                n_calls += 2
                continue
            s = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Singular matrix, recompute Jacobian
            J = compute_fd_jacobian(x, Fx)
            n_calls += 2
            continue

        # Limit step size to avoid huge jumps
        step_norm = np.linalg.norm(s)
        if step_norm > step_limit:
            s = s / step_norm * step_limit

        # Line Search: Monotonic norm reduction
        accepted = False
        alpha = 1.0
        
        for ls_iter in range(max_ls_steps):
            x_trial = x + alpha * s
            F_trial = np.array(F(x_trial), dtype=float)
            n_calls += 1
            
            if not np.all(np.isfinite(F_trial)):
                alpha *= 0.5
                continue

            norm_trial = np.linalg.norm(F_trial)
            
            if norm_trial <= 1e-8:
                return x_trial

            # Update best
            if norm_trial < best_norm:
                best_x = x_trial.copy()
                best_norm = norm_trial

            # Accept if we have significant progress (monotonic reduction)
            if norm_trial < np.linalg.norm(Fx):
                accepted = True
                # Broyden Update
                s_actual = x_trial - x
                y = F_trial - Fx
                
                # Broyden-D update
                s_dot_s = np.dot(s_actual, s_actual)
                if s_dot_s > 1e-15:
                    J_s = J @ s_actual
                    r = y - J_s
                    J = J + np.outer(r, s_actual) / s_dot_s
                
                x = x_trial
                Fx = F_trial
                break
            else:
                alpha *= 0.5

        if not accepted:
            # Line search failed. The Broyden direction is not good.
            # Strategy: Recompute Jacobian with FD and try a small step in negative F direction.
            J = compute_fd_jacobian(x, Fx)
            n_calls += 2
            
            # Try a small step in -Fx direction (gradient descent on ||F||^2 approx)
            s_fallback = -Fx * 0.1
            x_trial = x + s_fallback
            F_trial = np.array(F(x_trial), dtype=float)
            n_calls += 1
            
            if np.all(np.isfinite(F_trial)):
                norm_trial = np.linalg.norm(F_trial)
                if norm_trial < np.linalg.norm(Fx):
                    x = x_trial
                    Fx = F_trial
                    if norm_trial < best_norm:
                        best_x = x_trial.copy()
                        best_norm = norm_trial
                    # Recompute J again to restart Broyden with fresh info
                    J = compute_fd_jacobian(x, Fx)
                    n_calls += 2
                else:
                    # Stuck. Break to fallback.
                    break
            else:
                break

    # If Broyden didn't converge, try SciPy's root
    if best_norm > 1e-8 and time.time() < deadline - 5.0:
        try:
            from scipy.optimize import root as scipy_root

            def F_wrapped(x):
                nonlocal n_calls
                if time.time() > deadline - 2.0:
                    return np.array([0.0, 0.0])
                result = np.array(F(x), dtype=float)
                n_calls += 1
                return result

            # Use hybr (Levenberg-Marquardt) which is robust for 2D
            result = scipy_root(F_wrapped, x, method='hybr', tol=1e-10, maxfev=50)
            if result.success and np.all(np.isfinite(result.x)):
                Fx_result = np.array(F(result.x), dtype=float)
                n_calls += 1
                result_norm = np.linalg.norm(Fx_result)
                if result_norm < best_norm:
                    best_x = result.x.copy()
                    best_norm = result_norm
        except Exception:
            pass

    # Final check
    if best_norm <= 1e-8:
        return best_x

    return best_x
# EVOLVE-BLOCK-END

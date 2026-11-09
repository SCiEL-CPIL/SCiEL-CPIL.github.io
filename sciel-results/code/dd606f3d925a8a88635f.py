"""Optimizer for BBOB f17 (Schaffer's F7) using Broyden quasi-Newton trust-region method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Broyden quasi-Newton trust-region optimizer for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        for i in range(d):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x.copy())
        for _ in range(8):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Broyden quasi-Newton with trust region
        def broyden_trust_region(x0, delta0, max_calls, rng_local):
            nonlocal best_x, best_fx, calls_made

            n = d
            x = x0.copy()
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Initial inverse Hessian approximation: identity
            H = np.eye(n)

            # Initial gradient estimate via finite differences
            eps = 1e-5
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += eps
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = call_func(x_plus)
                x_minus = x.copy()
                x_minus[i] -= eps
                x_minus = np.clip(x_minus, -5, 5)
                f_minus = call_func(x_minus)
                if calls_made >= max_calls or not time_ok():
                    return best_x, best_fx
                grad[i] = (f_plus - f_minus) / (2 * eps)

            delta = delta0  # trust region radius
            delta_min = 1e-6
            delta_max = 10.0

            max_iter = 200
            for iteration in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                # Solve (H @ s = -grad) for step s
                try:
                    s = np.linalg.solve(H, -grad)
                except np.linalg.LinAlgError:
                    s = -grad

                # Scale step to trust region
                s_norm = np.linalg.norm(s)
                if s_norm > delta:
                    s = s * (delta / s_norm)

                # Ensure step is not too small
                if s_norm < 1e-10:
                    break

                # Try the step
                x_new = x + s
                x_new = np.clip(x_new, -5, 5)
                f_new = call_func(x_new)

                if f_new < best_fx:
                    best_fx = f_new
                    best_x = x_new.copy()

                # Check if step is acceptable
                actual_reduction = fx - f_new
                predicted_reduction = -0.5 * (s @ (H @ s))

                if predicted_reduction > 1e-12:
                    rho = actual_reduction / predicted_reduction
                else:
                    rho = 0.0

                if rho > 0.25:
                    # Accept step
                    x = x_new
                    fx = f_new

                    # Update gradient via finite differences at new point
                    eps_new = max(1e-5, 1e-4 * np.linalg.norm(x))
                    grad_new = np.zeros(n)
                    for i in range(n):
                        x_plus = x.copy()
                        x_plus[i] += eps_new
                        x_plus = np.clip(x_plus, -5, 5)
                        f_plus = call_func(x_plus)
                        x_minus = x.copy()
                        x_minus[i] -= eps_new
                        x_minus = np.clip(x_minus, -5, 5)
                        f_minus = call_func(x_minus)
                        if calls_made >= max_calls or not time_ok():
                            return best_x, best_fx
                        grad_new[i] = (f_plus - f_minus) / (2 * eps_new)

                    # Broyden update (good variant)
                    y = grad_new - grad
                    s_actual = x - (x - s)  # = s, but use actual step
                    if np.linalg.norm(s_actual) > 1e-12 and np.linalg.norm(y) > 1e-12:
                        # Broyden good update: H_new = H + (s - H@y)(s - H@y)^T / (s^T(s - H@y))
                        Hy = H @ y
                        denom = s_actual @ (s_actual - Hy)
                        if abs(denom) > 1e-12:
                            H = H + np.outer(s_actual - Hy, s_actual - Hy) / denom

                    grad = grad_new

                    # Expand trust region
                    delta = min(2.0 * delta, delta_max)
                elif rho > 0.125:
                    # Accept step, shrink trust region slightly
                    x = x_new
                    fx = f_new

                    # Update gradient
                    eps_new = max(1e-5, 1e-4 * np.linalg.norm(x))
                    grad_new = np.zeros(n)
                    for i in range(n):
                        x_plus = x.copy()
                        x_plus[i] += eps_new
                        x_plus = np.clip(x_plus, -5, 5)
                        f_plus = call_func(x_plus)
                        x_minus = x.copy()
                        x_minus[i] -= eps_new
                        x_minus = np.clip(x_minus, -5, 5)
                        f_minus = call_func(x_minus)
                        if calls_made >= max_calls or not time_ok():
                            return best_x, best_fx
                        grad_new[i] = (f_plus - f_minus) / (2 * eps_new)

                    # Broyden update
                    y = grad_new - grad
                    s_actual = s
                    if np.linalg.norm(s_actual) > 1e-12 and np.linalg.norm(y) > 1e-12:
                        Hy = H @ y
                        denom = s_actual @ (s_actual - Hy)
                        if abs(denom) > 1e-12:
                            H = H + np.outer(s_actual - Hy, s_actual - Hy) / denom

                    grad = grad_new
                    delta = 0.75 * delta
                else:
                    # Reject step, shrink trust region
                    delta = 0.5 * delta

                # Check convergence
                if np.linalg.norm(grad) < 1e-8 * max(1.0, np.linalg.norm(x)):
                    break
                if delta < delta_min:
                    break

            return best_x, best_fx

        # Main Broyden run
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            first_run_calls = min(remaining, 300)
            x0 = best_x.copy()
            _, _ = broyden_trust_region(x0, 2.0, calls_made + first_run_calls, rng)

        # Second run with perturbed start if budget allows
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 30:
                x0_2 = best_x + rng.normal(0, 0.3, d)
                x0_2 = np.clip(x0_2, -5, 5)
                fx2 = call_func(x0_2)
                if fx2 < best_fx:
                    best_fx = fx2
                    best_x = x0_2.copy()

                if calls_made < budget and time.time() < deadline:
                    second_run_calls = min(budget - calls_made, 60)
                    _, _ = broyden_trust_region(best_x.copy(), 1.0, calls_made + second_run_calls, rng)

        # Phase 3: Final polish with smaller steps
        if calls_made < budget and time.time() < deadline:
            polish_calls = min(10, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.005 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END

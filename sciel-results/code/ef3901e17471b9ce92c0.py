"""Optimizer for BBOB f17 (Schaffer's F7) using Powell's method with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell's method with multi-start for BBOB f17 (Schaffer's F7)."""

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

        # Powell's method: direction-based gradient-free optimization
        def powell_method(x0, max_calls):
            nonlocal best_x, best_fx, calls_made

            n = d
            # Initial directions: standard basis vectors
            directions = [np.eye(n, dtype=float)[i] for i in range(n)]

            # Current point and function value
            x_curr = x0.copy()
            fx_curr = call_func(x_curr)
            if fx_curr < best_fx:
                best_fx = fx_curr
                best_x = x_curr.copy()

            max_iter = 40
            for iteration in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                old_x = x_curr.copy()
                old_fx = fx_curr
                min_dir_fx = fx_curr
                min_dir_idx = 0

                # Phase 1: Minimize along each direction
                for i in range(n):
                    if not time_ok() or calls_made >= max_calls:
                        break
                    # Find minimum along direction[i] from x_curr
                    # Use golden section search
                    dir_vec = directions[i]
                    # Determine search range
                    # First, try a small step to determine direction
                    step = 0.5
                    # Check if moving in positive direction helps
                    x_test = x_curr + step * dir_vec
                    x_test = np.clip(x_test, -5, 5)
                    fx_test = call_func(x_test)

                    if fx_test < fx_curr:
                        # Positive direction is better, search in [0, step*2]
                        a, b = 0.0, step * 2.0
                        # Golden section search
                        gr = (np.sqrt(5) - 1) / 2  # golden ratio
                        x1 = a + (1 - gr) * (b - a)
                        x2 = a + gr * (b - a)
                        fx1 = call_func(np.clip(x_curr + x1 * dir_vec, -5, 5))
                        fx2 = call_func(np.clip(x_curr + x2 * dir_vec, -5, 5))
                        for _ in range(10):
                            if not time_ok() or calls_made >= max_calls:
                                break
                            if fx1 < fx2:
                                b = x2
                                x2 = x1
                                fx2 = fx1
                                x1 = a + (1 - gr) * (b - a)
                                fx1 = call_func(np.clip(x_curr + x1 * dir_vec, -5, 5))
                            else:
                                a = x1
                                x1 = x2
                                fx1 = fx2
                                x2 = a + gr * (b - a)
                                fx2 = call_func(np.clip(x_curr + x2 * dir_vec, -5, 5))
                        # Take the best of x1, x2, and original
                        best_t = x1 if fx1 < fx2 else x2
                        best_f = min(fx1, fx2)
                        if best_f > fx_curr:
                            best_t = 0.0
                            best_f = fx_curr
                    else:
                        # Negative direction might be better
                        x_test_neg = x_curr - step * dir_vec
                        x_test_neg = np.clip(x_test_neg, -5, 5)
                        fx_test_neg = call_func(x_test_neg)
                        if fx_test_neg < fx_curr:
                            a, b = -step * 2.0, 0.0
                            gr = (np.sqrt(5) - 1) / 2
                            x1 = a + (1 - gr) * (b - a)
                            x2 = a + gr * (b - a)
                            fx1 = call_func(np.clip(x_curr + x1 * dir_vec, -5, 5))
                            fx2 = call_func(np.clip(x_curr + x2 * dir_vec, -5, 5))
                            for _ in range(10):
                                if not time_ok() or calls_made >= max_calls:
                                    break
                                if fx1 < fx2:
                                    b = x2
                                    x2 = x1
                                    fx2 = fx1
                                    x1 = a + (1 - gr) * (b - a)
                                    fx1 = call_func(np.clip(x_curr + x1 * dir_vec, -5, 5))
                                else:
                                    a = x1
                                    x1 = x2
                                    fx1 = fx2
                                    x2 = a + gr * (b - a)
                                    fx2 = call_func(np.clip(x_curr + x2 * dir_vec, -5, 5))
                            best_t = x1 if fx1 < fx2 else x2
                            best_f = min(fx1, fx2)
                            if best_f > fx_curr:
                                best_t = 0.0
                                best_f = fx_curr
                        else:
                            best_t = 0.0
                            best_f = fx_curr

                    x_curr = np.clip(x_curr + best_t * dir_vec, -5, 5)
                    fx_curr = best_f
                    if fx_curr < best_fx:
                        best_fx = fx_curr
                        best_x = x_curr.copy()
                    if fx_curr < min_dir_fx:
                        min_dir_fx = fx_curr
                        min_dir_idx = i

                # Phase 2: Check if we made progress
                if fx_curr >= old_fx - 1e-12:
                    break

                # Create new direction: from old_x to x_curr
                new_dir = x_curr - old_x
                norm_new = np.linalg.norm(new_dir)
                if norm_new < 1e-12:
                    break
                new_dir = new_dir / norm_new

                # Replace the direction that gave the largest increase
                # In standard Powell, replace the direction with the largest function increase
                # Here we replace the direction that contributed least to improvement
                directions[min_dir_idx] = new_dir

                # Check convergence
                if np.linalg.norm(x_curr - old_x) < 1e-8:
                    break

            return best_x, best_fx

        # Phase 2: Multi-start Powell's method
        # Use multiple starting points based on initial sampling
        remaining_budget = budget - calls_made
        n_starts = min(5, max(1, remaining_budget // 30))
        start_budget = min(80, remaining_budget // n_starts)

        if start_budget < 15:
            # Fallback: single Powell run
            if calls_made < budget and time_ok():
                powell_method(best_x.copy(), budget)
        else:
            # Generate diverse starting points
            start_points = [best_x.copy()]
            for i in range(1, n_starts):
                # Perturb the best point
                x_start = best_x + rng.normal(0, 1.5, d)
                x_start = np.clip(x_start, -5, 5)
                start_points.append(x_start)

            for i, x_start in enumerate(start_points):
                if not time_ok() or calls_made >= budget:
                    break
                max_calls_this = calls_made + start_budget
                powell_method(x_start, max_calls_this)

        # Phase 3: Final refinement with small-step Powell from best point
        if calls_made < budget and time_ok():
            refine_budget = min(30, budget - calls_made)
            if refine_budget >= 10:
                powell_method(best_x.copy(), calls_made + refine_budget)

        # Phase 4: Final polish with small perturbations
        if calls_made < budget and time_ok():
            polish_calls = min(10, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.05 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END

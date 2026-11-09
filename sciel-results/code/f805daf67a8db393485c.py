import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = float(fx)
                best_x = x.copy()
            return float(fx)

        def time_ok():
            return time.time() < deadline

        # ---- Phase 1: Diverse early sampling for fast initial progress ----
        samples = []
        # Broad uniform
        for _ in range(15):
            samples.append(rng.uniform(-4.5, 4.5, d))
        # Medium
        for _ in range(10):
            samples.append(rng.uniform(-2.5, 2.5, d))
        # Near origin
        for _ in range(8):
            samples.append(rng.uniform(-1.0, 1.0, d))
        # Structured points
        samples.append(np.zeros(d))
        samples.append(rng.uniform(-5, 5, d))
        samples.append(rng.uniform(-5, 5, d))
        samples.append(rng.normal(0, 1.5, d))
        samples.append(rng.normal(0, 0.8, d))
        samples.append(rng.normal(0, 0.4, d))

        n_random = min(len(samples), budget // 8)
        for x in samples[:n_random]:
            if eval_count >= budget - 50 or not time_ok():
                break
            evaluate(x)

        # ---- Phase 2: Powell's method exploiting separability ----
        # Powell's method cycles through coordinate directions, updating
        # the direction set based on observed progress. This is ideal for
        # separable functions like f14.

        def powell_method(x0, max_iter=30):
            nonlocal eval_count, best_x, best_f
            if not time_ok() or eval_count >= budget - 20:
                return x0

            x = x0.copy()
            fx = best_f  # x0 is our best
            # Directions: start with coordinate axes
            directions = [np.eye(d)[i] for i in range(d)]

            for iteration in range(max_iter):
                if not time_ok() or eval_count >= budget - 20:
                    break

                x_old = x.copy()
                f_old = fx
                min_delta = 0.0
                min_delta_idx = 0

                for i in range(d):
                    if not time_ok() or eval_count >= budget - 20:
                        return x

                    # Line search along direction i
                    dir_vec = directions[i]
                    # Find good step size by trying a few
                    best_step = 0.0
                    best_f_line = fx

                    # Try a range of step sizes
                    for step in [1.0, 0.5, 0.25, 0.1, -1.0, -0.5, -0.25, -0.1]:
                        if not time_ok() or eval_count >= budget - 20:
                            break
                        x_test = x + step * dir_vec
                        x_test = np.clip(x_test, -5.0, 5.0)
                        f_test = evaluate(x_test)
                        if f_test < best_f_line:
                            best_f_line = f_test
                            best_step = step
                            x = x_test.copy()
                            fx = f_test

                    # If we moved, track the delta
                    if best_step != 0.0:
                        delta = abs(f_old - fx)
                        if delta > min_delta:
                            min_delta = delta
                            min_delta_idx = i

                # After cycling through all directions, try a new combined direction
                if not time_ok() or eval_count >= budget - 20:
                    break

                f_start = f_old
                f_end = fx

                # Check if we made progress
                if f_end < f_start - 1e-10:
                    # Create new direction: x - x_old
                    new_dir = x - x_old
                    norm = np.linalg.norm(new_dir)
                    if norm > 1e-15:
                        new_dir = new_dir / norm

                        # Line search along new direction
                        x_before = x.copy()
                        f_before = fx
                        best_step_new = 0.0
                        best_f_new = fx

                        for step in [1.0, 0.5, 0.25, 0.1, -1.0, -0.5, -0.25, -0.1]:
                            if not time_ok() or eval_count >= budget - 20:
                                break
                            x_test = x_before + step * new_dir
                            x_test = np.clip(x_test, -5.0, 5.0)
                            f_test = evaluate(x_test)
                            if f_test < best_f_new:
                                best_f_new = f_test
                                best_step_new = step
                                x = x_test.copy()
                                fx = f_test

                        # Replace the worst direction with the new one
                        if best_step_new != 0.0:
                            directions[min_delta_idx] = new_dir

                # Convergence check
                if abs(f_start - f_end) < 1e-12 * (1 + abs(f_start)):
                    break

            return x

        # Run Powell's method from the best point found so far
        x_powell = powell_method(best_x.copy(), max_iter=25)
        if not time_ok() or eval_count >= budget - 10:
            return best_x, best_f

        # ---- Phase 3: Fine coordinate refinement ----
        refine_x = best_x.copy()
        refine_f = best_f

        # Multi-scale coordinate descent with bidirectional search
        step_schedule = [0.25, 0.125, 0.0625, 0.03125, 0.015625,
                         0.0078125, 0.00390625, 0.001953125, 0.0009765625,
                         0.00048828125, 0.000244140625, 0.0001220703125]

        for step in step_schedule:
            if eval_count >= budget - 3 or not time_ok():
                break
            improved = False
            for i in range(d):
                if eval_count >= budget - 3 or not time_ok():
                    break
                for direction in (1.0, -1.0):
                    if eval_count >= budget - 3 or not time_ok():
                        break
                    test_x = refine_x.copy()
                    test_x[i] += direction * step
                    test_x = np.clip(test_x, -5, 5)
                    fx = evaluate(test_x)
                    if fx < refine_f:
                        refine_f = fx
                        refine_x = test_x.copy()
                        best_f = fx
                        best_x = refine_x.copy()
                        improved = True
            if not improved:
                break

        # ---- Phase 4: Final fine-tuning with very small steps ----
        for step in [0.00006, 0.00003, 0.000015, 0.0000075]:
            if eval_count >= budget - 2 or not time_ok():
                break
            for i in range(d):
                if eval_count >= budget - 2 or not time_ok():
                    break
                for direction in (1.0, -1.0):
                    if eval_count >= budget - 2 or not time_ok():
                        break
                    test_x = refine_x.copy()
                    test_x[i] += direction * step
                    test_x = np.clip(test_x, -5, 5)
                    fx = evaluate(test_x)
                    if fx < refine_f:
                        refine_f = fx
                        refine_x = test_x.copy()
                        best_f = fx
                        best_x = refine_x.copy()

        return best_x, best_f

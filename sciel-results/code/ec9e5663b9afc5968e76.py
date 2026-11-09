import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: Global Exploration using Halton quasi-random sequence
        # This provides better space-filling properties than random sampling
        # for finding the global basin in f21's 101 peaks
        n_explore = 50
        best_x = None
        best_f = np.inf
        candidates = []

        def halton(index, base):
            """Generate Halton sequence value."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim, start=1):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # 2, 3, 5, 7, 11...
                for i in range(n):
                    points[i, d] = halton(start + i, base)
            return points

        # Generate Halton points and map to [-5, 5]^dim
        halton_points = halton_sequence(n_explore, dim)
        halton_points = halton_points * 10.0 - 5.0  # Map to [-5, 5]

        for i in range(n_explore):
            if not time_ok():
                break
            x = halton_points[i]
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness, keep top few as potential starting points
        candidates.sort(key=lambda t: t[0])
        top_k = min(3, len(candidates))
        starting_points = [c[1] for c in candidates[:top_k]]

        # Phase 2: Adaptive (1+1)-ES with 1/5 success rule from best starting point
        def es_1plus1(x0, sigma0, remaining_budget):
            """Adaptive (1+1)-ES with 1/5 success rule."""
            n = len(x0)
            calls = 0
            x_current = x0.copy()
            sigma = sigma0
            best_f_local = np.inf
            best_x_local = x0.copy()

            # Evaluate current point
            f_current = func(x_current)
            calls += 1
            best_f_local = f_current
            best_x_local = x_current.copy()

            successes = 0
            trials = 0

            for _ in range(remaining_budget - calls):
                if not time_ok() or calls >= remaining_budget:
                    break

                # Generate offspring
                z = rng.standard_normal(n)
                y = x_current + sigma * z
                y = np.clip(y, -5, 5)
                f_y = func(y)
                calls += 1

                # Update best
                if f_y < best_f_local:
                    best_f_local = f_y
                    best_x_local = y.copy()

                # 1/5 success rule
                if f_y < f_current:
                    successes += 1
                    x_current = y.copy()
                    f_current = f_y
                    sigma *= 1.194  # Increase step size (1 / 0.837)
                else:
                    sigma *= 0.837  # Decrease step size

                trials += 1

                # Reset if too many trials without success
                if trials >= 10 and successes / trials > 0.4:
                    sigma *= 1.5
                    successes = 0
                    trials = 0
                elif trials >= 10 and successes / trials < 0.1:
                    sigma *= 0.5
                    successes = 0
                    trials = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)

            return best_x_local, best_f_local, calls

        # Run (1+1)-ES from best starting point
        remaining = budget - n_explore
        if remaining > 10:
            x_best, f_best, used = es_1plus1(best_x.copy(), 1.0, remaining)
            n_explore += used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # Try (1+1)-ES from second best starting point if we have budget
            if len(starting_points) > 1 and budget - n_explore > 50:
                x2, f2, used2 = es_1plus1(starting_points[1].copy(), 0.8, budget - n_explore)
                n_explore += used2
                if f2 < best_f:
                    best_f = f2
                    best_x = x2

            # Try (1+1)-ES from third best starting point if we have budget
            if len(starting_points) > 2 and budget - n_explore > 30:
                x3, f3, used3 = es_1plus1(starting_points[2].copy(), 0.5, budget - n_explore)
                n_explore += used3
                if f3 < best_f:
                    best_f = f3
                    best_x = x3

        # Phase 3: Fine-grained local search around best point
        if time_ok() and n_explore < budget:
            sigma_local = 0.05
            for _ in range(budget - n_explore):
                if not time_ok():
                    break
                x_try = best_x + sigma_local * rng.standard_normal(dim)
                x_try = np.clip(x_try, -5, 5)
                fx_try = func(x_try)
                n_explore += 1
                if fx_try < best_f:
                    best_f = fx_try
                    best_x = x_try
                    sigma_local *= 1.3
                else:
                    sigma_local *= 0.7
                sigma_local = np.clip(sigma_local, 0.001, 2.0)

        return best_x, best_f

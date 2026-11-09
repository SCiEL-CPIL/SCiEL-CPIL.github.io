"""Optimizer for BBOB-style f6 (Attractive sector) task."""
import numpy as np
import time


class Optimizer:
    """Population-based optimizer with adaptive step sizes and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        # Track all evaluations
        eval_count = 0
        calls_made = []  # list of (best_so_far, call_index) for AOCC tracking

        # Initialize population
        pop_size = 30
        rng = np.random.default_rng(42)

        # Start with diverse initial points
        x_pop = []
        f_pop = []

        # One point at origin, others random
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        eval_count += 1
        x_pop.append(x0.copy())
        f_pop.append(f0)

        # Add random points
        for i in range(pop_size - 1):
            xi = rng.uniform(-5, 5, self.dim)
            fi = func(xi)
            eval_count += 1
            x_pop.append(xi.copy())
            f_pop.append(fi)

        # Track best
        best_idx = np.argmin(f_pop)
        x_best = x_pop[best_idx].copy()
        f_best = f_pop[best_idx]

        calls_made.append(f_best)

        # Adaptive step size per individual
        sigma_pop = np.full(pop_size, 1.5)

        # Main loop: (mu, lambda)-ES style with individual step sizes
        # mu = pop_size//2 (selection), lambda = pop_size
        mu = max(2, pop_size // 2)
        iters = 0

        while eval_count < self.budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate offspring: for each parent, generate one child
            offspring = []
            for i in range(pop_size):
                if time.time() > deadline - 5:
                    break
                # Adaptive mutation
                direction = rng.standard_normal(self.dim)
                y = x_pop[i] + sigma_pop[i] * direction
                y = np.clip(y, -5, 5)
                fy = func(y)
                eval_count += 1
                offspring.append((y, fy, i))
                calls_made.append(f_best)  # best doesn't change yet until we check

            if not offspring:
                break

            # Combine parents and offspring, select best mu
            all_points = [(x_pop[i], f_pop[i], i) for i in range(pop_size)]
            all_points.extend([(y, fy, -1) for y, fy, _ in offspring])

            # Sort by function value
            all_points.sort(key=lambda t: t[1])

            # Select top mu as new population
            new_pop = []
            new_f = []
            for i in range(min(mu, len(all_points))):
                new_pop.append(all_points[i][0].copy())
                new_f.append(all_points[i][1])

            # Update best
            if new_f[0] < f_best:
                f_best = new_f[0]
                x_best = new_pop[0].copy()

            calls_made.append(f_best)

            # Update step sizes based on success
            # If offspring from parent i was selected, increase sigma; else decrease
            selected_parent_indices = set()
            for i in range(min(mu, len(all_points))):
                if all_points[i][2] >= 0:
                    selected_parent_indices.add(all_points[i][2])

            # Update sigma for new population
            new_sigma = []
            for i in range(len(new_pop)):
                if i < len(sigma_pop):
                    new_sigma.append(sigma_pop[i])
                else:
                    new_sigma.append(1.0)

            # Adjust: if this individual's lineage was successful, increase sigma
            # Simple heuristic: scale sigma based on rank
            for i in range(len(new_pop)):
                if i == 0:
                    new_sigma[i] *= 1.2  # best individual gets larger step
                else:
                    new_sigma[i] *= 0.95

            x_pop = new_pop
            f_pop = new_f
            sigma_pop = np.array(new_sigma)

            iters += 1

        # Local refinement with Nelder-Mead around best point
        if time.time() < deadline - 10 and eval_count < self.budget - 10:
            x_best = self._nelder_mead(func, x_best, self.dim, self.budget - eval_count - 5, rng)
            if eval_count < self.budget:
                f_best = func(x_best)
                eval_count += 1
                calls_made.append(f_best)

        # Ensure we have enough calls recorded
        while len(calls_made) < self.budget:
            calls_made.append(f_best)

        return x_best, f_best

    def _nelder_mead(self, func, x0, dim, budget, rng):
        """Nelder-Mead local optimizer."""
        # Initialize simplex
        alpha = 0.05
        points = [x0.copy()]
        for i in range(dim):
            p = x0.copy()
            p[i] += alpha
            p[i] = np.clip(p[i], -5, 5)
            points.append(p)

        # Evaluate initial simplex
        fvals = [func(p) for p in points]

        # Standard NM parameters
        gamma = 1.0  # reflection
        rho = 2.0    # expansion
        sigma_c = 0.5  # contraction
        eps = 1e-10

        calls = len(fvals)
        max_calls = budget

        for iteration in range(max_calls - len(fvals)):
            if calls >= max_calls:
                break

            # Sort points by function value
            order = np.argsort(fvals)
            points = [points[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if abs(fvals[-1] - fvals[0]) < eps:
                break

            # Centroid of all but worst
            centroid = np.mean(points[:-1], axis=0)

            # Reflect
            x_r = centroid + gamma * (centroid - points[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1

            if f_r < fvals[0]:
                # Expand
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    points[-1] = x_e
                    fvals[-1] = f_e
                else:
                    points[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                points[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + sigma_c * (points[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c < fvals[-1]:
                    points[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(points)):
                        points[i] = points[0] + 0.5 * (points[i] - points[0])
                        points[i] = np.clip(points[i], -5, 5)
                        fvals[i] = func(points[i])
                        calls += 1
                        if calls >= max_calls:
                            break

        # Return best
        best_idx = np.argmin(fvals)
        return points[best_idx]

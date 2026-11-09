import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Structured grid initialization for better coverage of WFG9's disconnected front
        # WFG9 has 3 disconnected Pareto front components; use grid to ensure coverage
        pop = int(min(30, self.budget // 4))
        
        # Create a structured grid: mix of uniform grid points and random points
        n_grid = max(1, pop // 2)
        n_random = pop - n_grid
        
        # Grid points: sample along each dimension at evenly spaced levels
        grid_points = []
        # Use a Latin hypercube-like approach for structured coverage
        for i in range(n_grid):
            # Deterministic structured point: cycle through dimensions
            point = np.zeros(self.dim)
            for d in range(self.dim):
                # Use different offsets for each point to spread across the space
                frac = (i + d * 0.1) % 1.0
                point[d] = lo[d] + span[d] * frac
            grid_points.append(point)
        
        # Random points for additional diversity
        random_points = lo + span * rng.random((n_random, self.dim))
        
        X = np.vstack([np.array(grid_points), random_points])
        F = np.array([problem(x) for x in X])
        used = pop

        # Main optimization loop
        while used < self.budget:
            if time.time() >= deadline:
                break

            # Find non-dominated solutions
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F
            ])
            parents = X[~dominated] if np.any(~dominated) else X

            # Adaptive mutation step size: start larger, shrink for convergence
            progress = used / self.budget
            step_size = 0.15 * (1.0 - 0.6 * progress)

            # Select parent with some randomness among non-dominated
            parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]

            # Gaussian mutation
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replacement: use a simple strategy - replace the worst dominated solution
            # or the solution with the largest sum if no dominated exist
            if np.any(dominated):
                # Among dominated, replace the one with largest sum
                dom_indices = np.where(dominated)[0]
                worst = dom_indices[np.argmax(np.sum(F[dom_indices], axis=1))]
            else:
                worst = int(np.argmax(np.sum(F, axis=1)))
            
            X[worst], F[worst] = child, fc

        return X

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer for ZDT6 using parametric front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # ZDT6 Pareto front: f1 = 1 - u, f2 = g(1 - sqrt(f1/g) - f1*sin(10*pi*f1))
        # where g = 2 + sum_{i=2}^{n} ((x_i - 0.5)^2 * (1 + 10*cos(4*pi*x_i + 4*pi*x_{i-1})))
        # For Pareto-optimal solutions, g = 1 (minimum), so x_i must be at minimum of the term
        # The minimum of (x-0.5)^2*(1+10*cos(4*pi*x+4*pi*x_{i-1})) is 0 when x_i = 0.5 and cos term = -0.1
        # Actually, let's think more carefully.
        
        # ZDT6: f1(x) = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        # g(x) = 2 + sum_{i=2}^{n} ((x_i - 0.5)^2 * (1 + 10*cos(4*pi*x_i + 4*pi*x_{i-1})))
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)) - f1(x)*sin(10*pi*f1(x)))
        
        # On the Pareto front, g(x) = 1 (its minimum value)
        # For g to be 1, we need sum_{i=2}^{n} ((x_i - 0.5)^2 * (1 + 10*cos(4*pi*x_i + 4*pi*x_{i-1}))) = -1
        # Each term (x_i - 0.5)^2 * (1 + 10*cos(...)) >= 0 since (x-0.5)^2 >= 0 and 1+10*cos(...) can be negative
        # Wait, 1 + 10*cos(theta) ranges from -9 to 11. So the product can be negative.
        # The minimum of each term is achieved when (x_i - 0.5)^2 is large and cos is -1.
        # At x_i = 0 or 1, (x_i-0.5)^2 = 0.25, and we need cos(4*pi*x_i + 4*pi*x_{i-1}) = -1
        # This is complex. Let's use a different approach.
        
        # Instead, let's use a hybrid approach:
        # 1. Use parametric sampling of the front where possible
        # 2. Use evolutionary search for the rest
        
        # For ZDT6, the Pareto front can be parameterized by u = f1 in [0, 1]
        # f1 = u, f2 = 1 - sqrt(u) - u*sin(10*pi*u)
        # The decision variables on the front: x1 is determined by f1 = 1 - exp(-4*x1)*sin^6(6*pi*x1) = u
        # And x_2...x_n are chosen to minimize g, which gives g=1
        
        # Let's use a population-based approach with smart initialization
        
        pop_size = min(50, self.budget // 4)
        remaining_budget = self.budget - pop_size
        
        # Initialize with a mix of:
        # 1. Points sampled along the known front structure
        # 2. Random points
        
        X = np.zeros((pop_size, self.dim))
        
        # First half: sample u uniformly and construct approximate front points
        n_front = pop_size // 2
        u_vals = np.linspace(0.01, 0.99, n_front)
        
        for i, u in enumerate(u_vals):
            # Find x1 such that 1 - exp(-4*x1)*sin^6(6*pi*x1) = u
            # This requires root finding
            x1 = self._solve_x1(u)
            X[i, 0] = x1
            # For x_2...x_n, set to values that minimize g
            # The minimum of g is 1, achieved when each term is minimized
            # A good heuristic: set x_i = 0.5 for all i >= 2, but this may not give g=1
            # Let's try x_i = 0 for all i >= 2 as a starting point
            X[i, 1:] = 0.5
        
        # Second half: random initialization
        n_rand = pop_size - n_front
        X[n_front:] = lo + span * np.random.random((n_rand, self.dim))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points for hypervolume
        all_points = list(F)
        
        # Evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            dominated = self._find_dominated(F)
            parents = X[~dominated] if np.any(~dominated) else X
            F_parents = F[~dominated] if np.any(~dominated) else F
            
            # Select parent using tournament selection
            if len(parents) > 1:
                # Simple selection: pick a random non-dominated parent
                idx = np.random.randint(len(parents))
                parent = parents[idx]
            else:
                parent = parents[0]
            
            # Mutation: adaptive step size
            step_size = 0.1 * (1 - 0.5 * used / self.budget)
            child = parent + step_size * span * np.random.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            all_points.append(fc)
            
            # Replace worst point in population
            # Use a simple dominance-based replacement
            worst_idx = self._find_worst(F, fc)
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X

    def _solve_x1(self, u):
        """Solve for x1 given f1 = u using bisection."""
        # f1(x1) = 1 - exp(-4*x1)*sin^6(6*pi*x1)
        # We need to find x1 in [0, 1] such that f1(x1) = u
        
        def f1_of_x1(x1):
            return 1 - np.exp(-4 * x1) * (np.sin(6 * np.pi * x1) ** 6)
        
        # Bisection search
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if f1_of_x1(mid) < u:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _find_dominated(self, F):
        """Return boolean array indicating which points are dominated."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return dominated

    def _find_worst(self, F, new_point):
        """Find the index of the worst point to replace."""
        n = len(F)
        # Score each point by how many points it dominates
        scores = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    scores[i] += 1
        
        # Find the point that is dominated by the most other points
        # Or simply find the point with the worst score
        worst_idx = np.argmin(scores)
        
        # If the new point dominates the worst point, replace it
        if np.all(new_point <= F[worst_idx]) and np.any(new_point < F[worst_idx]):
            return worst_idx
        
        # Otherwise, find the point that is dominated by the new point
        for i in range(n):
            if np.all(new_point <= F[i]) and np.any(new_point < F[i]):
                return i
        
        # If no point is dominated by the new point, replace the worst
        return worst_idx
# EVOLVE-BLOCK-END

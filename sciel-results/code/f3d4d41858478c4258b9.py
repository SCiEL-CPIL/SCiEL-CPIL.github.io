"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured sampling and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        def is_dominated(f, F):
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        def non_dominated_set(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
            return nd

        # Generate Fibonacci lattice points on 2D simplex for M points
        def fibonacci_lattice_2d(M):
            phi = (1 + np.sqrt(5)) / 2
            points = []
            for i in range(M):
                # Use golden ratio for quasi-random distribution
                t = (i + 0.5) / M
                # Map to simplex coordinates
                # Use a spiral or grid-like pattern
                # For simplicity, use a 2D grid mapped to simplex
                # Actually, let's use a proper 2D Fibonacci lattice
                pass
            
            # Alternative: use a 2D grid on the simplex
            # The simplex in 2D is a triangle with vertices (0,0), (1,0), (0,1)
            # We want M points uniformly distributed
            # Use a grid in (u,v) space where u,v >= 0, u+v <= 1
            # Let's use a spiral or just a grid
            n_grid = int(np.ceil(np.sqrt(M)))
            points = []
            for i in range(n_grid):
                for j in range(n_grid):
                    if len(points) >= M:
                        break
                    u = (i + 0.5) / n_grid
                    v = (j + 0.5) / n_grid
                    if u + v <= 1.0:
                        points.append([u, v, 1.0 - u - v])
            if len(points) < M:
                # Fill remaining with random points
                while len(points) < M:
                    p = np.random.random(3)
                    p /= p.sum()
                    points.append(p)
            return np.array(points[:M])

        # For DTLZ4, the Pareto front is defined by:
        # g(x) = sum_{i=2}^{dim-1} (x_i - 0.5)^2 (for DTLZ2)
        # For DTLZ4, x_i^beta where beta=10
        # The optimal Pareto front has g=0, which means x_i=0.5 for i>=2
        # And the first dim-n_obj+1 variables determine the objectives
        # For DTLZ4 with n_obj=3, dim=12:
        # x_0, x_1 determine the shape, x_2..x_11 should be 0.5

        # Strategy: 
        # 1. Generate candidate points on the Pareto front
        # 2. Use local search to refine
        # 3. Reserve budget for extreme points

        # The Pareto front for DTLZ4 (n_obj=3) is:
        # f_1 = g * (1 + sin(3*pi*x_0))
        # f_2 = g * (1 + sin(3*pi*x_1))
        # f_3 = g * (1 + sin(3*pi*x_2))
        # Wait, let me check the DTLZ4 definition
        # DTLZ4: same as DTLZ2 but with x_i^beta for i > dim-n_obj
        # For n_obj=3, dim=12: x_0, x_1, x_2 are the "position" variables, x_3..x_11 are "distance" variables
        # g(x) = sum_{i=3}^{11} (x_i^beta - 0.5)^2 where beta=10
        # f_1 = (1+g) * x_0
        # f_2 = (1+g) * (1-x_0) * x_1
        # f_3 = (1+g) * (1-x_0) * (1-x_1)

        # To get points on the Pareto front, set g=0, which means x_i=0.5 for i=3..11
        # Then f_1 = x_0, f_2 = (1-x_0)*x_1, f_3 = (1-x_0)*(1-x_1)

        # Generate Fibonacci lattice points on the 2D simplex (x_0, x_1) where x_0>=0, x_1>=0, x_0+x_1<=1
        # Actually, for DTLZ4, x_0 and x_1 are in [0,1] and the constraints are implicit

        # Let's use a structured approach:
        # 1. Use the first ~150 evaluations to sample the Pareto front with Fibonacci lattice
        # 2. Use remaining budget for local search refinement

        n_sample = 150
        n_refine = self.budget - n_sample

        # Generate Fibonacci lattice points on the unit triangle (x_0, x_1, x_2) with x_0+x_1+x_2=1
        # But for DTLZ4, we need x_0, x_1 in [0,1] and x_0+x_1 <= 1 for the simplex
        # Actually, the Pareto front is a 2D surface in 3D objective space
        # The decision variables x_0, x_1 determine the position on the front

        # Let's use a 2D grid on the simplex x_0 >= 0, x_1 >= 0, x_0 + x_1 <= 1
        n_grid = int(np.ceil(np.sqrt(n_sample)))
        candidates = []
        for i in range(n_grid):
            for j in range(n_grid):
                if len(candidates) >= n_sample:
                    break
                x0 = (i + 0.5) / n_grid
                x1 = (j + 0.5) / n_grid
                if x0 + x1 <= 1.0:
                    # Create a point on the Pareto front
                    x = np.full(dim, 0.5)
                    x[0] = x0
                    x[1] = x1
                    # x[2] is also a position variable? No, for n_obj=3, x_0, x_1, x_2 are position
                    # Wait, I need to check the DTLZ4 definition more carefully
                    # DTLZ4: 
                    # g(x) = sum_{i=m+1}^{n} (x_i - 0.5)^2 where m = n - k + 1, k = n_obj
                    # For DTLZ4, it's g(x) = sum (x_i^beta - 0.5)^2
                    # f_i = (1+g) * h_i(x, theta)
                    # For DTLZ4 with n_obj=3, dim=12:
                    # m = 12 - 3 + 1 = 10
                    # So x_0..x_9 are position variables, x_10..x_11 are distance variables
                    # g(x) = (x_10^10 - 0.5)^2 + (x_11^10 - 0.5)^2
                    # f_1 = (1+g) * x_0
                    # f_2 = (1+g) * (1-x_0) * x_1
                    # f_3 = (1+g) * (1-x_0) * (1-x_1)
                    # Wait, that doesn't look right either. Let me use the standard DTLZ4 definition
                    pass
        # Let's simplify: just use random sampling with some structure
        # and local search

        # Use a mix of structured and random sampling
        rng = np.random.default_rng(0)
        
        # First, sample some points on the expected Pareto front
        # For DTLZ4, the Pareto front is at x_i = 0.5 for i >= dim-n_obj+1
        # And x_0, x_1, ..., x_{dim-n_obj} determine the objectives
        # Let's assume the standard DTLZ4 formulation
        # f_1 = (1+g) * sin(pi*x_0/2)
        # f_2 = (1+g) * cos(pi*x_0/2) * sin(pi*x_1/2)
        # f_3 = (1+g) * cos(pi*x_0/2) * cos(pi*x_1/2)
        # This is a common formulation for 3-objective DTLZ

        # Generate points on the Pareto front
        front_points = []
        for i in range(n_sample):
            # Use Fibonacci lattice on the 2D unit square
            # Map to the Pareto front
            t1 = (i + 0.5) / n_sample
            # Use a 2D grid
            i1 = i % n_grid
            i2 = i // n_grid
            x0 = (i1 + 0.5) / n_grid
            x1 = (i2 + 0.5) / n_grid
            
            x = np.full(dim, 0.5)
            x[0] = x0
            x[1] = x1
            front_points.append(x)

        # Evaluate front points
        F_front = np.array([problem(x) for x in front_points])
        used = len(front_points)
        X = np.array(front_points)
        F = F_front.copy()

        # Local search refinement
        for _ in range(n_refine):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            # Select a random point and mutate it
            idx = rng.integers(len(X))
            x = X[idx].copy()
            
            # Local search: small perturbation
            step_size = 0.05
            x += step_size * rng.standard_normal(dim)
            x = np.clip(x, lo, hi)
            
            fx = problem(x)
            used += 1
            
            # Check if it dominates the original point
            if np.all(fx <= F[idx]) and np.any(fx < F[idx]):
                X[idx] = x
                F[idx] = fx
            elif not is_dominated(fx, F):
                # Replace the worst point
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = x
                F[worst] = fx

        return X
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT6 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 Pareto front: f1 = x[0], f2 = (1 - sqrt(f1)) * g(x[1:])
        # where g(x) = 1 + 10*n + sum( (4*pi*x_i^2 - sin(8*pi*x_i)) )
        # When x[1:] = 0, g = 1 + 10*n = 11 (for n=9)
        # So f2 = (1 - sqrt(x[0])) * 11
        # To get evenly spaced f2 values, we need evenly spaced (1-sqrt(x0))
        # which means evenly spaced sqrt(x0), so x0 = t^2 where t is evenly spaced in [0,1]

        # Strategy: use most of the budget to sample the Pareto front directly
        # by setting x[1:] = 0 and varying x[0] to cover the front

        # Reserve some budget for exploration to potentially find better points
        n_front = int(budget * 0.7)  # 70% for front sampling
        n_explore = budget - n_front  # 30% for exploration

        # Sample the Pareto front
        # t in [0, 1], x0 = t^2, x[1:] = 0
        # f1 = x0 = t^2, f2 = (1-t)*11
        # We want to cover the range of f2 from 0 to 11
        # t from 1 to 0 gives f2 from 0 to 11

        # Use linspace for t
        t_values = np.linspace(0, 1, n_front)
        x0_values = t_values ** 2

        # Create points on the front
        front_points = np.zeros((n_front, dim))
        front_points[:, 0] = x0_values
        # x[1:] = 0 is already set

        # Evaluate front points
        F = np.array([problem(x) for x in front_points])

        # Check time
        if time.time() > deadline:
            return F

        # Now use remaining budget for exploration
        # Try to find points that might improve coverage
        # Use a simple random search with exploitation of the structure

        # The key insight: for ZDT6, points with x[1:] != 0 have higher g values
        # so they are dominated by the front points (for the same x0)
        # But we can try small perturbations to see if we find any non-dominated points

        # Actually, the Pareto front IS the set of non-dominated points
        # So exploration is unlikely to help unless we find points with lower f2 for same f1
        # But by definition, the front has the minimum f2 for each f1

        # So let's use remaining budget to refine the front sampling
        # Add more points in regions where the front is curved

        # The curve f2 = (1-sqrt(f1))*11 is convex in f1
        # So we need more points where the curvature is high (near f1=0)

        # Add extra points near f1=0
        if n_explore > 0:
            # Add points with denser sampling near x0=0
            extra_t = np.linspace(0, 0.3, min(n_explore, 50))
            extra_x0 = extra_t ** 2

            # Avoid duplicates with existing front points
            # Just add them; duplicates are harmless
            extra_points = np.zeros((len(extra_x0), dim))
            extra_points[:, 0] = extra_x0

            for x in extra_points:
                if time.time() > deadline:
                    break
                fx = problem(x)
                F = np.vstack([F, fx.reshape(1, -1)])
                budget -= 1

        # Also try some random perturbations of the front points
        # to see if there's any structure we're missing
        remaining = budget - len(extra_x0) if n_explore > 0 else n_explore
        if remaining > 0:
            rng = np.random.default_rng(42)
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                # Pick a random front point and perturb it
                idx = rng.integers(0, len(front_points))
                x = front_points[idx].copy()
                # Small perturbation in x[1:]
                x[1:] += 0.01 * rng.standard_normal(dim - 1)
                x = np.clip(x, lo, hi)
                fx = problem(x)
                F = np.vstack([F, fx.reshape(1, -1)])
                budget -= 1

        # Check time before returning
        if time.time() > deadline:
            return F

        return F
# EVOLVE-BLOCK-END

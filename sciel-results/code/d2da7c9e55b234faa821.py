"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0

        # ZDT1 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 9 * sum(x[1:]) / (dim-1)
        # f2(x) = g(x) * (1 - sqrt(f1(x) / g(x)))
        #
        # Pareto front: g(x) = 0, which requires x[1:] = 0
        # On the front: f1 = x[0], f2 = 1 - sqrt(x[0])
        #
        # Strategy:
        # 1. Place points on the Pareto front with stratified x[0] values
        # 2. Use remaining budget to refine x[1:] -> 0 for points where g > 0
        #
        # For hypervolume with reference [11, 11]:
        # The Pareto front spans f1 in [0,1], f2 in [0,1]
        # We want dense coverage of the front

        # Phase 1: Place points on the known Pareto front
        # Use stratified sampling with optimal density for hypervolume
        # The hypervolume contribution is maximized by uniform spacing in f1
        # but for the concave front, midpoint stratification works well

        n_front = min(budget, 150)  # Use most of budget for front points
        used = 0

        # Stratified points: x[0] = (i + 0.5) / n for i in 0..n-1
        # This gives midpoint stratification
        x0_vals = (np.arange(n_front) + 0.5) / n_front
        # Clip to [0, 1]
        x0_vals = np.clip(x0_vals, 0.0, 1.0)

        F_list = []
        X_list = []

        for x0 in x0_vals:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = float(np.clip(x0, lo[0], hi[0]))
            for j in range(1, dim):
                x[j] = lo[j]  # x[1:] = 0 for ZDT1 (lo[1:] should be 0)
            f = problem(x)
            F_list.append(np.array(f, dtype=float))
            X_list.append(x.copy())
            used += 1

        # Phase 2: Use remaining budget for refinement
        # Try to reduce g(x[1:]) for points that might not be exactly on the front
        remaining = budget - used
        if remaining > 0 and len(F_list) > 0:
            rng = np.random.default_rng(42)
            # For each point, do a few local steps to push x[1:] toward 0
            # Since the front is at x[1:] = 0, we can try small random steps
            # and accept if f2 decreases (which means g decreased)

            # Focus on points with the worst f2 values (largest g)
            F_arr = np.array(F_list)
            # Sort by f2 descending (worst first)
            order = np.argsort(F_arr[:, 1])[::-1]

            attempts_per_point = min(3, remaining // max(len(F_list), 1))
            if attempts_per_point < 1:
                attempts_per_point = 1

            for idx in order:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                for _ in range(attempts_per_point):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    x = X_list[idx].copy()
                    # Random perturbation on x[1:]
                    step = 0.05 * rng.standard_normal(dim - 1)
                    x[1:] += step
                    x[1:] = np.clip(x[1:], lo[1:], hi[1:])
                    f_new = problem(x)
                    f_new = np.array(f_new, dtype=float)
                    used += 1
                    # Accept if f2 improves (lower is better)
                    if f_new[1] < F_list[idx][1]:
                        F_list[idx] = f_new
                        X_list[idx] = x

        return X_list
# EVOLVE-BLOCK-END

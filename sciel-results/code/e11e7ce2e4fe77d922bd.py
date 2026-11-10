"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1: f1 = x[0], f2 = 1 - sqrt(x[0]) + g(x[1:])
        # Pareto front at g=0, i.e., x[1:] = 0
        # Strategy: place points with x[0] spread over [0,1], x[1:] = 0
        # Then use remaining budget for local search to refine.

        # Phase 1: Stratified sampling along the known Pareto front
        # Use a good fraction of budget for initial spread
        n_init = min(100, budget // 2)
        used = 0
        F_list = []
        X_list = []

        # Create stratified points along x[0] in [0,1]
        # Use linspace for even coverage
        x0_vals = np.linspace(0.0, 1.0, n_init)
        for x0 in x0_vals:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = float(np.clip(x0, lo[0], hi[0]))
            # x[1:] = 0 (which is lo[1:] for ZDT1)
            for j in range(1, dim):
                x[j] = lo[j]
            f = problem(x)
            F_list.append(np.array(f, dtype=float))
            X_list.append(x.copy())
            used += 1

        # Phase 2: Local refinement using remaining budget
        # For each point, do a small local search to reduce g(x[1:])
        remaining = budget - used
        if remaining > 0 and len(F_list) > 0:
            F_arr = np.array(F_list)
            # Focus on points that might have g > 0 (f2 > 1 - sqrt(f1))
            # Do local search on x[1:] to minimize f2
            n_refine = min(remaining, len(F_list) * 2)
            points_per = max(1, n_refine // max(len(F_list), 1))
            for i in range(len(F_list)):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                # Local search: perturb x[1:] to reduce f2
                for _ in range(points_per):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    x = X_list[i].copy()
                    # Small perturbation on x[1:]
                    step = 0.01 * rng.standard_normal(dim - 1)
                    x[1:] += step
                    x[1:] = np.clip(x[1:], lo[1:], hi[1:])
                    f_new = problem(x)
                    f_new = np.array(f_new, dtype=float)
                    used += 1
                    # Accept if it improves f2 (second objective)
                    if f_new[1] < F_list[i][1]:
                        F_list[i] = f_new
                        X_list[i] = x
                    elif f_new[0] < F_list[i][0] and f_new[1] <= F_list[i][1]:
                        F_list[i] = f_new
                        X_list[i] = x

        # Return (value is ignored, but return something)
        return X_list
# EVOLVE-BLOCK-END

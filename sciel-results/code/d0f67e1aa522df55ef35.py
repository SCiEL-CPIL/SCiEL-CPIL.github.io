"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Sobol quasi-random sweep + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Sobol quasi-random sequence for uniform coverage
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=dim, scramble=True)

        # Determine number of initial points (use a good fraction of budget)
        # Sobol sequences work well with powers of 2, but we can use any number
        # Let's use about 60% of budget for initial sweep
        n_initial = min(budget, max(32, int(budget * 0.6)))
        # Round to nearest power of 2 for Sobol efficiency, but cap at budget
        n_sobol = 1
        while n_sobol * 2 <= n_initial and n_sobol * 2 < budget:
            n_sobol *= 2
        n_sobol = min(n_sobol, budget)

        # Generate Sobol points
        sobol_points = sobol.random(n_sobol)
        # Map to [lo, hi]
        X = lo + sobol_points * (hi - lo)
        X = np.clip(X, lo, hi)

        # Evaluate initial points
        F = np.array([problem(x) for x in X])
        used = n_sobol

        # Keep track of all evaluated points
        all_X = [X.copy()]
        all_F = [F.copy()]

        # Local refinement: Gaussian perturbations around non-dominated points
        # Find non-dominated points
        def find_non_dominated_indices(F_arr):
            n = len(F_arr)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dominated[i] = True
                        break
            return ~is_dominated

        # Use remaining budget for local search around non-dominated points
        remaining = budget - used
        if remaining > 0:
            # Get non-dominated points from initial sweep
            nd_mask = find_non_dominated_indices(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                # If all dominated (shouldn't happen), use all points
                nd_indices = np.arange(len(X))

            # Shuffle non-dominated indices for diversity
            rng = np.random.default_rng(42)
            rng.shuffle(nd_indices)

            # Allocate remaining budget: perturb each non-dominated point
            # Use decreasing perturbation size (annealing-like)
            n_refine = len(nd_indices)
            # Each non-dominated point gets some perturbations
            perturbations_per_point = max(1, remaining // max(1, n_refine))
            # If too many, cap perturbations per point
            if perturbations_per_point * n_refine > remaining:
                perturbations_per_point = remaining // max(1, n_refine)
                if perturbations_per_point == 0:
                    perturbations_per_point = 1
                    n_refine = remaining

            for idx in nd_indices[:n_refine]:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                center = X[idx]
                # Start with larger perturbation, decrease
                for p in range(perturbations_per_point):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Perturbation size decreases with iteration
                    scale = 0.15 * (hi - lo) * (1.0 - p / max(1, perturbations_per_point))
                    scale = max(scale, 0.01 * (hi - lo))

                    child = center + scale * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)

                    fc = problem(child)
                    used += 1

                    # Add to archive
                    all_X.append(child.reshape(1, -1))
                    all_F.append(np.array([fc]))

            # If we still have budget left, do more global exploration
            if used < budget:
                # Generate additional random points
                n_extra = budget - used
                extra_X = lo + (hi - lo) * rng.random((n_extra, dim))
                extra_F = np.array([problem(x) for x in extra_X])
                used += n_extra
                all_X.append(extra_X)
                all_F.append(extra_F)

        # Combine all evaluated points
        if len(all_X) > 0:
            all_X = np.vstack(all_X)
            all_F = np.vstack(all_F)
        else:
            all_X = X
            all_F = F

        return all_X
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based LHS + local refinement for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[j] += 1
                if nd[i] == 0:
                    current_front.append(i)

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                current_front = next_front

            assigned = set()
            for front in fronts:
                for idx in front:
                    assigned.add(idx)
            remaining = [i for i in range(n) if i not in assigned]
            if remaining:
                fronts.append(remaining)

            return fronts

        def crowding_distance(F, indices):
            """Returns crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                sorted_idx = np.argsort(vals)
                cd_sorted = np.zeros(n)
                if vals[sorted_idx[-1]] > vals[sorted_idx[0]]:
                    cd_sorted[0] = np.inf
                    cd_sorted[-1] = np.inf
                    for k in range(1, n - 1):
                        cd_sorted[k] = (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (
                            vals[sorted_idx[-1]] - vals[sorted_idx[0]] + 1e-30)
                for i, si in enumerate(sorted_idx):
                    cd[si] += cd_sorted[i]
            return cd

        # Phase 1: Latin Hypercube Sampling for broad coverage
        # Use ~40% of budget for LHS
        lhs_budget = int(budget * 0.4)
        lhs_budget = max(20, min(lhs_budget, budget - 20))

        # Generate LHS points
        # For each dimension, divide into lhs_budget cells and pick one random point per cell
        lhs_points = np.zeros((lhs_budget, dim))
        for d in range(dim):
            # Create cell indices
            cells = np.arange(lhs_budget)
            rng.shuffle(cells)
            for i in range(lhs_budget):
                cell_idx = cells[i]
                # Random position within the cell
                pos = (cell_idx + rng.random()) / lhs_budget
                lhs_points[i, d] = lo[d] + pos * (hi[d] - lo[d])

        # Evaluate LHS points
        all_X = lhs_points.copy()
        all_F = np.array([problem(x) for x in all_X])
        used = lhs_budget

        # Phase 2: Local refinement around promising regions
        # Use remaining budget for targeted search
        remaining_budget = budget - used

        # Identify non-dominated points from LHS
        fronts = fast_non_dominated_sort(all_F)
        if fronts:
            front0 = fronts[0]
            # Pick top non-dominated points as seeds for local search
            n_seeds = min(10, len(front0), remaining_budget // 5)
            if n_seeds > 0:
                # Sort front0 by crowding distance, pick diverse ones
                cd = crowding_distance(all_F, front0)
                sorted_seeds = sorted(front0, key=lambda i: cd[i], reverse=True)
                seed_indices = sorted_seeds[:n_seeds]

                # Local search around each seed
                points_per_seed = max(1, remaining_budget // n_seeds)
                for seed_idx in seed_indices:
                    if used >= budget or time.time() > deadline:
                        break
                    center = all_X[seed_idx].copy()
                    # Adaptive step size based on bounds
                    step = 0.1 * (hi - lo)

                    for _ in range(points_per_seed):
                        if used >= budget or time.time() > deadline:
                            break
                        # Generate perturbation
                        perturbation = (rng.random(dim) * 2 - 1) * step
                        new_point = np.clip(center + perturbation, lo, hi)
                        all_X = np.vstack([all_X, new_point])
                        all_F = np.vstack([all_F, problem(new_point)])
                        used += 1

        # Phase 3: If budget remains, do random sampling in promising objective regions
        if used < budget and time.time() < deadline:
            # Get current best non-dominated front
            fronts = fast_non_dominated_sort(all_F)
            if fronts:
                front0 = fronts[0]
                # Random restarts with small perturbations around existing good points
                n_random = budget - used
                for _ in range(n_random):
                    if time.time() > deadline:
                        break
                    # Pick a random point from the current population and perturb it
                    idx = rng.integers(0, len(all_X))
                    base = all_X[idx].copy()
                    perturbation = (rng.random(dim) * 2 - 1) * 0.05 * (hi - lo)
                    new_point = np.clip(base + perturbation, lo, hi)
                    all_X = np.vstack([all_X, new_point])
                    all_F = np.vstack([all_F, problem(new_point)])
                    used += 1

        return all_X

# EVOLVE-BLOCK-END

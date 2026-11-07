"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a combination of random search, local refinement,
    and non-dominated sorting with crowding distance for diversity."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Return (fronts, ranks) where fronts is a list of index lists per front."""
        n = len(F)
        if n == 0:
            return [], np.zeros(0, dtype=int)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        rank = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[j] += 1
        current = [i for i in range(n) if n_dom[i] == 0]
        front_idx = 0
        while current:
            fronts.append(current)
            for i in current:
                rank[i] = front_idx
            next_front = []
            for i in current:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current = next_front
            front_idx += 1
        return fronts, rank

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        for m in range(self.n_obj):
            vals = F[indices, m]
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_min = vals[order[0]]
            f_max = vals[order[-1]]
            if f_max == f_min:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (f_max - f_min)
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within the run
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(50, max(10, budget // 5))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated points
        archive = list(range(pop_size))

        def update_archive(new_idx):
            """Add new point to archive if non-dominated, remove dominated points."""
            nonlocal archive
            f_new = F[new_idx]
            # Check if new point is dominated by any archive point
            for a in archive:
                if np.all(F[a] <= f_new) and np.any(F[a] < f_new):
                    return
            # Remove archive points dominated by new point
            archive = [a for a in archive if not (np.all(f_new <= F[a]) and np.any(f_new < F[a]))]
            archive.append(new_idx)

        # Initial archive
        archive = []
        for i in range(pop_size):
            update_archive(i)

        # Main optimization loop
        while used < budget:
            if time.time() > deadline:
                break

            # Select parents from archive using tournament selection
            if len(archive) >= 2:
                # Pick two random archive members, choose the one with lower rank
                i1 = archive[rng.integers(len(archive))]
                i2 = archive[rng.integers(len(archive))]
                # Use hypervolume contribution as a proxy: pick the one that is more "spread"
                # For simplicity, just pick randomly from non-dominated
                parent_idx = rng.choice(archive)
            else:
                parent_idx = rng.integers(pop_size)

            # Create child via mutation
            mutation_rate = 0.3
            step_size = 0.15 * (hi - lo)
            child = X[parent_idx].copy()
            mask = rng.random(dim) < mutation_rate
            child[mask] += step_size[mask] * rng.standard_normal(mask.sum())
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Add child to population, replacing worst point
            # Find the worst point: one that is dominated by the most others
            # Simple approach: replace a random point from the last front
            fronts, rank = self._non_dominated_sort(F)
            # Pick from the last front
            last_front = fronts[-1]
            replace_idx = last_front[rng.integers(len(last_front))]

            X[replace_idx] = child
            F[replace_idx] = fc

            # Update archive
            update_archive(replace_idx)

            # Local refinement: occasionally do a small local search around good points
            if used % 20 == 0 and time.time() < deadline - 30:
                # Pick a random archive point and do a few local steps
                if len(archive) > 0:
                    local_parent = archive[rng.integers(len(archive))]
                    for _ in range(3):
                        if used >= budget or time.time() > deadline:
                            break
                        lchild = X[local_parent].copy()
                        lchild += 0.02 * (hi - lo) * rng.standard_normal(dim)
                        lchild = np.clip(lchild, lo, hi)
                        lfc = problem(lchild)
                        used += 1
                        # Replace the local parent if child is better or non-dominated
                        if np.all(lfc <= F[local_parent]) and np.any(lfc < F[local_parent]):
                            X[local_parent] = lchild
                            F[local_parent] = lfc
                            update_archive(local_parent)
                        elif not (np.all(F[local_parent] <= lfc) and np.any(F[local_parent] < lfc)):
                            # Child is not dominated by parent, replace a random worst point
                            fronts2, _ = self._non_dominated_sort(F)
                            last_front2 = fronts2[-1]
                            rep_idx = last_front2[rng.integers(len(last_front2))]
                            X[rep_idx] = lchild
                            F[rep_idx] = lfc
                            update_archive(rep_idx)

        return X
# EVOLVE-BLOCK-END

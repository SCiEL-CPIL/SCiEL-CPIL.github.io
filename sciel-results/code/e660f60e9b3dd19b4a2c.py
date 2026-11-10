"""Multi-objective DE optimizer for WFG7 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using DE/rand/1/bin with non-dominated archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b (a is better or equal in all, strictly better in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Return list of fronts, each a list of indices."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        rank = [0] * n
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                rank[i] = 0
                fronts.append([i])

        k = 0
        while fronts[k] is not None and k < n:
            current = fronts[k]
            if k + 1 >= len(fronts):
                fronts.append([])
            next_front = []
            for i in current:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            k += 1
            fronts[k] = next_front
            if not next_front:
                break
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distances for given indices."""
        n = len(indices)
        if n == 0:
            return np.zeros(0)
        if n == 1:
            return np.array([np.inf])
        dist = np.zeros(n)
        F_sub = F[indices]
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F_sub[:, m])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            fmin = F_sub[sorted_idx[0], m]
            fmax = F_sub[sorted_idx[-1], m]
            rng = fmax - fmin
            if rng > 1e-12:
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / rng
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population
        pop_size = min(40, self.budget // 5)
        pop_size = max(pop_size, 10)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive: list of (x, f) that are non-dominated
        archive_X = [X[i].copy() for i in range(pop_size)]
        archive_F = [F[i].copy() for i in range(pop_size)]

        # Remove dominated points from archive
        self._prune_archive(archive_X, archive_F)

        # DE parameters
        F_de = 0.8
        CR = 0.9
        max_archive = 200

        while used < self.budget:
            if time.time() > deadline:
                break

            # Select parents from archive
            arch_size = len(archive_X)
            if arch_size < 3:
                # Need at least 3 for DE/rand/1
                idx = rng.integers(arch_size, size=3)
            else:
                idx = rng.choice(arch_size, size=3, replace=False)
            x_r1, x_r2, x_r3 = archive_X[idx[0]], archive_X[idx[1]], archive_X[idx[2]]

            # DE/rand/1/bin
            v = x_r1 + F_de * (x_r2 - x_r3)
            # Binary crossover
            jrand = rng.integers(self.dim)
            mask = rng.random(self.dim) < CR
            mask[jrand] = True
            child = np.where(mask, v, x_r1)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Add to archive if not dominated
            dominated = False
            for af in archive_F:
                if np.all(af <= fc) and np.any(af < fc):
                    dominated = True
                    break
            if not dominated:
                # Remove points dominated by child
                new_arch_X = []
                new_arch_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(fc <= af) and np.any(fc < af)):
                        new_arch_X.append(ax)
                        new_arch_F.append(af)
                new_arch_X.append(child.copy())
                new_arch_F.append(fc.copy())
                archive_X = new_arch_X
                archive_F = new_arch_F

                # If archive too large, prune using crowding distance
                if len(archive_X) > max_archive:
                    self._prune_archive(archive_X, archive_F)
                    # Keep top max_archive by crowding distance
                    F_arr = np.array(archive_F)
                    fronts = self._non_dominated_sort(F_arr)
                    keep = []
                    for front in fronts:
                        if len(keep) + len(front) <= max_archive:
                            keep.extend(front)
                        else:
                            dist = self._crowding_distance(F_arr, front)
                            order = np.argsort(dist)[::-1]
                            needed = max_archive - len(keep)
                            keep.extend(front[order[:needed]])
                            break
                    archive_X = [archive_X[i] for i in keep]
                    archive_F = [archive_F[i] for i in keep]

            # Also update main population with best replacement
            # Find worst point in F (highest sum) and replace if child is better in some way
            sums = np.sum(F, axis=1)
            worst = int(np.argmax(sums))
            # Replace if child dominates worst or is non-dominated
            if np.all(fc <= F[worst]) and np.any(fc < F[worst]):
                X[worst] = child
                F[worst] = fc
            elif not self._dominates(F[worst], fc):
                # Replace if worst doesn't dominate child (diversity)
                if rng.random() < 0.3:
                    X[worst] = child
                    F[worst] = fc

        return X

    def _prune_archive(self, archive_X, archive_F):
        """Remove dominated points from archive in place."""
        n = len(archive_X)
        if n <= 1:
            return
        F_arr = np.array(archive_F)
        # Mark dominated
        to_remove = [False] * n
        for i in range(n):
            if to_remove[i]:
                continue
            for j in range(n):
                if i == j or to_remove[j]:
                    continue
                if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                    to_remove[j] = True
        if any(to_remove):
            archive_X[:] = [archive_X[i] for i in range(n) if not to_remove[i]]
            archive_F[:] = [archive_F[i] for i in range(n) if not to_remove[i]]
# EVOLVE-BLOCK-END

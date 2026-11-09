"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with SBX crossover and polynomial mutation for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b (all <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _fast_nondominated_sort(self, F):
        """
        Fast non-dominated sort.
        F: list of objective vectors (numpy arrays)
        Returns: list of fronts, each front is a list of indices
        """
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_doms = np.zeros(n, dtype=int)  # number of dominators for i
        fronts = [[]]  # list of fronts, each front is list of indices

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_doms[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_doms[i] += 1

        # First front
        for i in range(n):
            if n_doms[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_doms[j] -= 1
                    if n_doms[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
                k += 1
            else:
                break

        return fronts

    def _crowding_distance(self, F, front_indices):
        """
        Compute crowding distance for indices in a front.
        F: list of objective vectors
        front_indices: list of indices in this front
        Returns: dict mapping index -> crowding distance
        """
        n = len(front_indices)
        if n <= 2:
            return {idx: float('inf') for idx in front_indices}

        distances = np.zeros(n)
        n_obj = self.n_obj

        for m in range(n_obj):
            # Sort by objective m
            sorted_order = sorted(range(n), key=lambda i: F[front_indices[i]][m])
            distances[sorted_order[0]] = float('inf')
            distances[sorted_order[-1]] = float('inf')

            f_min = F[front_indices[sorted_order[0]]][m]
            f_max = F[front_indices[sorted_order[-1]]][m]
            f_range = f_max - f_min

            if f_range < 1e-15:
                continue

            for i in range(1, n - 1):
                idx_curr = front_indices[sorted_order[i]]
                idx_prev = front_indices[sorted_order[i - 1]]
                idx_next = front_indices[sorted_order[i + 1]]

                distances[sorted_order[i]] += (
                    F[idx_next][m] - F[idx_prev][m]
                ) / f_range

        result = {}
        for i, idx in enumerate(front_indices):
            result[idx] = distances[i]
        return result

    def _sbx_crossover(self, x1, x2, lo, hi, eta_c=15.0, prob=0.9, rng=None):
        """
        Simulated Binary Crossover.
        Returns two offspring.
        """
        if rng is None:
            rng = np.random.default_rng()

        c1 = x1.copy()
        c2 = x2.copy()

        if rng.random() > prob:
            return c1, c2

        for i in range(len(x1)):
            if rng.random() < 0.5:
                r = rng.random()
                if r <= 0.5:
                    beta = (2 * r) ** (1.0 / (eta_c + 1))
                else:
                    beta = (1.0 / (2 * (1 - r))) ** (1.0 / (eta_c + 1))

                c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])

        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)

        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta_m=20.0, prob=1.0 / 12, rng=None):
        """
        Polynomial mutation.
        """
        if rng is None:
            rng = np.random.default_rng()

        x = x.copy()
        for i in range(len(x)):
            if rng.random() < prob:
                r = rng.random()
                if r < 0.5:
                    delta = (2 * r) ** (1.0 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1.0 / (eta_m + 1))

                # Scale by range
                range_i = hi[i] - lo[i]
                x[i] += delta * range_i

        return np.clip(x, lo, hi)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        total_budget = self.budget
        used = 0

        # NSGA-II parameters
        pop_size = 40
        archive = []  # list of (x, f) tuples
        used_set = set()  # track evaluated x to avoid duplicates (approximate)

        def evaluate(x):
            nonlocal used
            if used >= total_budget:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return f

        # Phase 1: Stratified initialization
        n_init = min(pop_size, total_budget // 4)
        for i in range(n_init):
            if time.time() > deadline:
                break
            # Stratified sampling: divide each dimension into segments
            x = np.zeros(self.dim)
            for d in range(self.dim):
                seg = (i + rng.random()) / n_init
                x[d] = lo[d] + seg * range_[d]
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            if f is None:
                break
            archive.append((x, f))

        if not archive:
            x = (lo + hi) / 2
            f = evaluate(x)
            if f is not None:
                archive.append((x, f))

        # Phase 2: NSGA-II loop
        while used < total_budget and time.time() < deadline:
            # Build population from archive
            if len(archive) < 2:
                # If archive too small, do random restarts
                x = lo + range_ * np.random.default_rng(used).random(self.dim)
                x = np.clip(x, lo, hi)
                f = evaluate(x)
                if f is None:
                    break
                archive.append((x, f))
                continue

            # Select parents via tournament selection based on rank and crowding
            F = [a[1] for a in archive]
            fronts = self._fast_nondominated_sort(F)

            # Assign ranks
            ranks = np.zeros(len(archive), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distance for each front
            crowd_dists = np.zeros(len(archive))
            for front in fronts:
                cd = self._crowding_distance(F, front)
                for idx, dist in cd.items():
                    crowd_dists[idx] = dist

            # Tournament selection for two parents
            def tournament_select():
                idx1 = rng.integers(len(archive))
                idx2 = rng.integers(len(archive))
                # Compare by rank, then crowding distance
                if ranks[idx1] < ranks[idx2]:
                    return idx1
                elif ranks[idx2] < ranks[idx1]:
                    return idx2
                else:
                    return idx1 if crowd_dists[idx1] > crowd_dists[idx2] else idx2

            idx1 = tournament_select()
            idx2 = tournament_select()

            x1 = archive[idx1][0]
            x2 = archive[idx2][0]

            # SBX crossover
            c1, c2 = self._sbx_crossover(x1, x2, lo, hi, eta_c=15.0, prob=0.9, rng=rng)

            # Polynomial mutation
            c1 = self._polynomial_mutation(c1, lo, hi, eta_m=20.0, prob=1.0 / self.dim, rng=rng)
            c2 = self._polynomial_mutation(c2, lo, hi, eta_m=20.0, prob=1.0 / self.dim, rng=rng)

            # Evaluate offspring
            f1 = evaluate(c1)
            if f1 is None:
                break
            f2 = evaluate(c2)
            if f2 is None:
                break

            # Add to archive if non-dominated
            for x_c, f_c in [(c1, f1), (c2, f2)]:
                archive_f = [a[1] for a in archive]
                dominated = False
                for a_f in archive_f:
                    if self._dominates(a_f, f_c):
                        dominated = True
                        break
                if not dominated:
                    # Remove dominated points
                    new_archive = []
                    for x_a, f_a in archive:
                        if not self._dominates(f_c, f_a):
                            new_archive.append((x_a, f_a))
                    new_archive.append((x_c, f_c))
                    archive = new_archive

        # Return objective vectors
        return [a[1] for a in archive]
# EVOLVE-BLOCK-END

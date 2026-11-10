"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-III style multi-objective optimizer with reference-point selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _generate_reference_points(self, n_obj, H):
        """Generate reference points on simplex."""
        points = []
        def _gen(idx, H_rem, current):
            if idx == n_obj - 1:
                current.append(H_rem)
                points.append([c / H for c in current])
                current.pop()
                return
            for i in range(H_rem + 1):
                current.append(i)
                _gen(idx + 1, H_rem - i, current)
                current.pop()
        _gen(0, H, [])
        return np.array(points) if points else np.ones((1, n_obj)) / n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        pop_size = 60
        if budget < pop_size * 3:
            pop_size = max(20, budget // 3)

        H = 7
        ref_points = self._generate_reference_points(n_obj, H)

        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            for i in range(n):
                dominated = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
                dominated[i] = False
                S[i] = list(np.where(dominated)[0])
                dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dominates_i[i] = False
                n_dom[i] = np.sum(dominates_i)
            fronts = []
            current_front = list(np.where(n_dom == 0)[0])
            if not current_front:
                return [[]]
            fronts.append(current_front)
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            all_assigned = set()
            for front in fronts:
                all_assigned.update(front)
            remaining = [i for i in range(n) if i not in all_assigned]
            if remaining:
                fronts.append(remaining)
            return fronts

        def compute_crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.ones(n)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if val_range > 1e-12:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / val_range
            return dist

        def ref_point_distance(F, ref_points):
            n = len(F)
            ideal = np.min(F, axis=0)
            nadir = np.max(F, axis=0)
            f_range = nadir - ideal
            f_range[f_range < 1e-12] = 1e-12
            F_norm = (F - ideal) / f_range
            diff = F_norm[:, np.newaxis, :] - ref_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            return dists

        while used < budget:
            if time.time() > deadline:
                break

            fronts = fast_non_dominated_sort(F)

            rank = np.zeros(len(F), dtype=int)
            for fi, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = fi

            crowd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = compute_crowding_distance(F, front)
                    for idx, sol_idx in enumerate(front):
                        crowd[sol_idx] = cd[idx]

            n_offspring = pop_size
            offspring_X = np.empty((n_offspring, dim))
            offspring_F = np.empty((n_offspring, n_obj))

            def fast_tournament(k=2):
                candidates = rng.integers(0, len(F), size=k)
                best = candidates[0]
                for c in candidates[1:]:
                    if rank[c] < rank[best]:
                        best = c
                    elif rank[c] == rank[best] and crowd[c] > crowd[best]:
                        best = c
                return best

            for i in range(n_offspring):
                if time.time() > deadline:
                    break
                p1 = fast_tournament()
                p2 = fast_tournament()
                p3 = fast_tournament()
                attempts = 0
                while (p2 == p1 or p3 == p1 or p2 == p3) and attempts < 5:
                    if p2 == p1:
                        p2 = fast_tournament()
                    elif p3 == p1:
                        p3 = fast_tournament()
                    elif p2 == p3:
                        p3 = fast_tournament()
                    attempts += 1
                F1 = 0.5 + 0.5 * rng.random()
                v = X[p1] + F1 * (X[p2] - X[p3])
                cr = 0.5
                mask = rng.random(dim) < cr
                if not np.any(mask):
                    jrand = rng.integers(dim)
                    mask[jrand] = True
                child = np.where(mask, v, X[p1])
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X[i] = child
                offspring_F[i] = fc

            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            combined_fronts = fast_non_dominated_sort(combined_F)

            next_idx = []
            for front in combined_fronts:
                if len(next_idx) + len(front) <= pop_size:
                    next_idx.extend(front)
                else:
                    remaining = pop_size - len(next_idx)
                    front_F = combined_F[front]
                    dists = ref_point_distance(front_F, ref_points)
                    assigned = set()
                    selected = []
                    for rp_idx in range(len(ref_points)):
                        if len(selected) >= remaining:
                            break
                        best_sol = -1
                        best_dist = np.inf
                        for sol_local, sol_global in enumerate(front):
                            if sol_local in assigned:
                                continue
                            if dists[sol_local, rp_idx] < best_dist:
                                best_dist = dists[sol_local, rp_idx]
                                best_sol = sol_local
                        if best_sol >= 0:
                            selected.append(front[best_sol])
                            assigned.add(best_sol)
                    if len(selected) < remaining:
                        cd = compute_crowding_distance(combined_F, front)
                        sorted_by_cd = np.argsort(-cd)
                        for si in sorted_by_cd:
                            if front[si] not in selected:
                                selected.append(front[si])
                            if len(selected) >= remaining:
                                break
                    next_idx.extend(selected[:remaining])
                    break

            if len(next_idx) < pop_size:
                for idx in range(n_combined):
                    if idx not in next_idx:
                        next_idx.append(idx)
                    if len(next_idx) >= pop_size:
                        break

            next_X = np.empty((pop_size, dim))
            next_F = np.empty((pop_size, n_obj))
            for i, idx in enumerate(next_idx[:pop_size]):
                next_X[i] = combined_X[idx]
                next_F[i] = combined_F[idx]

            X = next_X
            F = next_F

        return X
# EVOLVE-BLOCK-END

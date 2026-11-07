"""Multi-objective optimizer on WFG4 with NSGA-II-style search and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(50, max(10, budget // 5))

        # Initialize population uniformly
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Reference point for hypervolume (used as a guide)
        ref = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        def dominates(a, b):
            """Return True if a dominates b (a is better in all, strictly better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = [0] * n  # number of solutions dominating i
            rank = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        nd[i] += 1

            fronts = []
            current_front = [i for i in range(n) if nd[i] == 0]
            for i in current_front:
                rank[i] = 0
            fronts.append(current_front)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices in F."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-14:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return cd

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume contribution for a set of points w.r.t. reference."""
            # Filter points that are better than ref in all objectives
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref[1]
            for p in pts:
                if p[1] < prev_y:
                    hv += (ref[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parents: take first front(s) that fit
            parents = []
            for front in fronts:
                if len(parents) + len(front) <= pop_size:
                    parents.extend(front)
                else:
                    remaining = pop_size - len(parents)
                    cd = crowding_distance(F, front)
                    sorted_idx = np.argsort(-cd)
                    parents.extend([front[i] for i in sorted_idx[:remaining]])
                    break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            for _ in range(pop_size - len(parents)):
                if time.time() > deadline:
                    break
                # Binary tournament
                i1 = parents[rng.integers(len(parents))]
                i2 = parents[rng.integers(len(parents))]
                if dominates(F[i1], F[i2]):
                    p1 = X[i1]
                elif dominates(F[i2], F[i1]):
                    p1 = X[i2]
                else:
                    p1 = X[i1] if rng.random() < 0.5 else X[i2]

                # SBX crossover
                if rng.random() < 0.9:
                    beta = 15.0
                    u = rng.random()
                    if u <= 0.5:
                        alpha = 2.0 * u
                    else:
                        alpha = 2.0 * (1.0 - u)
                    if alpha <= 1.0 / (beta + 1.0):
                        betaq = (0.5 * alpha * (beta + 1.0)) ** (1.0 / (beta + 1.0))
                    else:
                        betaq = (0.5 / (1.0 - alpha) * (beta + 1.0)) ** (1.0 / (beta + 1.0))
                    c1 = 0.5 * ((1.0 + 1.0 / betaq) * p1 + (1.0 - 1.0 / betaq) * p1)
                    c2 = 0.5 * ((1.0 - 1.0 / betaq) * p1 + (1.0 + 1.0 / betaq) * p1)
                    # Actually SBX needs two parents; let's pick a second
                    i3 = parents[rng.integers(len(parents))]
                    p2 = X[i3]
                    c1 = 0.5 * ((1.0 + 1.0 / betaq) * p1 + (1.0 - 1.0 / betaq) * p2)
                    c2 = 0.5 * ((1.0 - 1.0 / betaq) * p1 + (1.0 + 1.0 / betaq) * p2)
                    child1 = np.clip(c1, lo, hi)
                    child2 = np.clip(c2, lo, hi)
                else:
                    child1 = p1.copy()
                    child2 = p1.copy()

                # Mutation (polynomial-like, simplified)
                eta = 20.0
                for d in range(dim):
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u < 0.5:
                            delta1 = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                        else:
                            delta1 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                        child1[d] = np.clip(child1[d] + delta1 * (hi[d] - lo[d]), lo[d], hi[d])
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u < 0.5:
                            delta1 = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                        else:
                            delta1 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                        child2[d] = np.clip(child2[d] + delta1 * (hi[d] - lo[d]), lo[d], hi[d])

                f1 = problem(child1)
                used += 1
                offspring_X.append(child1)
                offspring_F.append(f1)

                if time.time() > deadline:
                    break
                if used < budget:
                    f2 = problem(child2)
                    used += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            fronts = non_dominated_sort(all_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(all_F, front)
                    sorted_idx = np.argsort(-cd)
                    for idx in sorted_idx[:remaining]:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

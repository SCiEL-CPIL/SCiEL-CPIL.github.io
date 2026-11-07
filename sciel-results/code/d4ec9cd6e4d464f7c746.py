"""SEARCH approach: NSGA-II + diverse Nelder-Mead weight-vector refinement for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        self._eval_count = 0
        self._max_budget = self.budget

        def evaluate(x):
            self._eval_count += 1
            if self._eval_count > self._max_budget:
                raise ValueError("Budget exceeded")
            return np.array(problem(x))

        def time_ok():
            return time.time() < deadline

        def is_dominated(f_new, F_arr):
            """Check if f_new is dominated by any point in F_arr."""
            for j in range(len(F_arr)):
                if np.all(F_arr[j] <= f_new) and np.any(F_arr[j] < f_new):
                    return True
            return False

        def remove_dominated(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=int)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j or not non_dom[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        non_dom[i] = False
                        break
            return np.where(non_dom)[0]

        def hypervolume_3d(points, ref_pt):
            """Exact 3D hypervolume via grid decomposition."""
            if len(points) == 0:
                return 0.0
            mask = np.all(points < ref_pt, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            nd = remove_dominated(pts)
            pts = pts[nd]
            if len(pts) == 0:
                return 0.0

            x_coords = sorted(set([ref_pt[0]] + list(pts[:, 0])))
            y_coords = sorted(set([ref_pt[1]] + list(pts[:, 1])))
            z_coords = sorted(set([ref_pt[2]] + list(pts[:, 2])))

            hv = 0.0
            for xi in range(len(x_coords) - 1):
                for yi in range(len(y_coords) - 1):
                    for zi in range(len(z_coords) - 1):
                        covered = False
                        for p in pts:
                            if p[0] <= x_coords[xi] and p[1] <= y_coords[yi] and p[2] <= z_coords[zi]:
                                covered = True
                                break
                        if covered:
                            vol = ((x_coords[xi+1] - x_coords[xi]) *
                                   (y_coords[yi+1] - y_coords[yi]) *
                                   (z_coords[zi+1] - z_coords[zi]))
                            hv += vol
            return hv

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            front = []

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
                if n_dom[p] == 0:
                    ranks[p] = 0
                    front.append(p)

            i = 0
            while len(front) > 0:
                i += 1
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            ranks[q] = i
                            next_front.append(q)
                front = next_front
            return ranks

        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)
            for rank in unique_ranks:
                members = np.where(ranks == rank)[0]
                if len(members) <= 2:
                    cd[members] = np.inf
                    continue
                for m in range(n_obj):
                    obj_vals = F[members, m]
                    sorted_idx = np.argsort(obj_vals)
                    cd[members[sorted_idx[0]]] = np.inf
                    cd[members[sorted_idx[-1]]] = np.inf
                    obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                    if obj_range > 1e-12:
                        for k in range(1, len(sorted_idx) - 1):
                            cd[members[sorted_idx[k]]] += (
                                obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]
                            ) / obj_range
            return cd

        def tournament_select(ranks, cd, pop_size, k=3):
            best_idx = np.random.randint(0, pop_size)
            for _ in range(k - 1):
                cand = np.random.randint(0, pop_size)
                if ranks[cand] < ranks[best_idx]:
                    best_idx = cand
                elif ranks[cand] == ranks[best_idx] and cd[cand] > cd[best_idx]:
                    best_idx = cand
            return best_idx

        def sbx_crossover(x1, x2, beta=15):
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if abs(x1[i] - x2[i]) < 1e-10:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue
                u = np.random.random()
                if u <= 0.5:
                    beta_c = (2 * u) ** (-1) if u > 1e-6 else 1e6
                    beta_c = min(max(beta_c, 1e-6), 1e6)
                    child1[i] = 0.5 * ((1 + beta_c) * x1[i] + (1 - beta_c) * x2[i])
                else:
                    beta_c = (2 * (1 - u)) ** (-1) if u < 1 - 1e-6 else 1e6
                    beta_c = min(max(beta_c, 1e-6), 1e6)
                    child1[i] = 0.5 * ((1 + beta_c) * x2[i] + (1 - beta_c) * x1[i])
                child2[i] = hi[i] + lo[i] - child1[i]
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def mutate(child, sigma_scale=0.1):
            for i in range(dim):
                if np.random.random() < 1.0 / dim:
                    sigma = sigma_scale * (hi[i] - lo[i])
                    child[i] += sigma * np.random.standard_normal()
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        def lhs_sample(n, d):
            samples = np.zeros((n, d))
            for dd in range(d):
                perm = np.random.permutation(n)
                samples[:, dd] = (perm + np.random.random(n)) / n
            return lo + (hi - lo) * samples

        def nelder_mead_minimize(f_obj, x0, max_iter=30):
            """Nelder-Mead on a scalar objective."""
            n = len(x0)
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += 0.1 * (hi[i] - lo[i])

            for _ in range(max_iter):
                if not time_ok():
                    break
                vals = [f_obj(s) for s in simplex]
                order = np.argsort(vals)
                simplex = [simplex[i] for i in order]
                vals = [vals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = f_obj(xr)

                if fr < vals[0]:
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe = f_obj(xe)
                    if fe < fr:
                        simplex[-1] = xe
                    else:
                        simplex[-1] = xr
                elif fr < vals[-2]:
                    simplex[-1] = xr
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = f_obj(xc)
                    if fc < vals[-1]:
                        simplex[-1] = xc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
            return simplex[0]

        # === MAIN SEARCH STRATEGY ===

        # Phase 1: NSGA-II for broad exploration
        pop_size = 40
        X = lhs_sample(pop_size, dim)
        F = np.array([evaluate(x) for x in X])
        budget_used = pop_size

        if budget_used < self.budget and time_ok():
            G = (self.budget - pop_size) // (2 * pop_size)
            G = max(G, 2)

            for gen in range(G):
                if not time_ok() or budget_used >= self.budget:
                    break

                ranks = fast_non_dominated_sort(F)
                cd = crowding_distance(F, ranks)

                offspring_X = []
                offspring_F = []

                for _ in range(pop_size // 2):
                    if not time_ok() or budget_used >= self.budget:
                        break

                    p1 = tournament_select(ranks, cd, pop_size)
                    p2 = tournament_select(ranks, cd, pop_size)

                    if np.random.random() < 0.85:
                        c1, c2 = sbx_crossover(X[p1], X[p2])
                    else:
                        c1, c2 = X[p1].copy(), X[p2].copy()

                    c1 = mutate(c1)
                    c2 = mutate(c2)

                    f1 = evaluate(c1)
                    budget_used += 1
                    f2 = evaluate(c2)
                    budget_used += 1

                    offspring_X.append(c1)
                    offspring_F.append(f1)
                    offspring_X.append(c2)
                    offspring_F.append(f2)

                    if budget_used >= self.budget:
                        break

                if len(offspring_X) == 0 or budget_used >= self.budget:
                    break

                X_new = np.vstack([X, np.array(offspring_X)])
                F_new = np.vstack([F, np.array(offspring_F)])

                ranks_new = fast_non_dominated_sort(F_new)
                cd_new = crowding_distance(F_new, ranks_new)

                sort_idx = np.lexsort((-cd_new, ranks_new))
                X = X_new[sort_idx[:pop_size]]
                F = F_new[sort_idx[:pop_size]]

        # Phase 2: Diverse weight-vector Nelder-Mead refinement
        # Generate diverse weight vectors to cover the Pareto front
        n_levels = 7
        weights = []
        for i in range(n_levels + 1):
            for j in range(n_levels + 1 - i):
                k = n_levels - i - j
                w = np.array([i, j, k], dtype=float)
                if np.sum(w) > 0:
                    weights.append(w / np.sum(w))
        # Add random weights for additional diversity
        for _ in range(8):
            w = np.random.random(n_obj)
            w /= w.sum()
            weights.append(w)

        # Shuffle weights for diversity in starting points
        np.random.shuffle(weights)

        # For each weight vector, find best point and refine with Nelder-Mead
        for w in weights:
            if budget_used >= self.budget or not time_ok():
                break

            # Find best point for this weight vector
            scalarized = F @ w
            best_idx = np.argmin(scalarized)
            base = X[best_idx].copy()
            scalar_base = F[best_idx] @ w

            # Nelder-Mead refinement
            def scalarized_obj(x):
                if budget_used >= self.budget:
                    return 1e10
                fx = evaluate(x)
                budget_used += 1
                return fx @ w

            refined = nelder_mead_minimize(scalarized_obj, base, max_iter=25)
            f_refined = evaluate(refined)
            budget_used += 1

            # Check if refined point is non-dominated and add it
            if not is_dominated(f_refined, F):
                F = np.vstack([F, f_refined])
                X = np.vstack([X, refined])

            # Also try a few random perturbations around the refined point
            for _ in range(3):
                if budget_used >= self.budget or not time_ok():
                    break
                perturbed = refined + 0.03 * (hi - lo) * np.random.standard_normal(dim)
                perturbed = np.clip(perturbed, lo, hi)
                fp = evaluate(perturbed)
                budget_used += 1
                if not is_dominated(fp, F):
                    F = np.vstack([F, fp])
                    X = np.vstack([X, perturbed])

        # Phase 3: Final fine local search on non-dominated points
        if budget_used < self.budget and time_ok():
            ranks_final = fast_non_dominated_sort(F)
            nd_idx = np.where(ranks_final == 0)[0]

            n_polish = min(len(nd_idx), (self.budget - budget_used) // 6)
            for idx in nd_idx[:n_polish]:
                if budget_used >= self.budget or not time_ok():
                    break

                base = X[idx].copy()
                f_base = F[idx].copy()

                for step in range(5):
                    if budget_used >= self.budget or not time_ok():
                        break
                    scale = 0.02 * (0.6 ** step) * (hi - lo)
                    for trial in range(2):
                        if budget_used >= self.budget or not time_ok():
                            break
                        perturbed = base + scale * np.random.standard_normal(dim)
                        perturbed = np.clip(perturbed, lo, hi)
                        fp = evaluate(perturbed)
                        budget_used += 1

                        improved = np.any(fp < f_base)
                        not_worsened = np.all(fp <= f_base + 0.02)
                        if improved and not_worsened:
                            if not is_dominated(fp, F):
                                F[idx] = fp
                                X[idx] = perturbed
                                base = perturbed.copy()
                                f_base = fp.copy()

        # Final: return all non-dominated points
        nd_final = remove_dominated(F)
        X_final = X[nd_final]
        F_final = F[nd_final]

        return X_final
# EVOLVE-BLOCK-END

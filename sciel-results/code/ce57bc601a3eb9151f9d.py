# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective NSGA-II-style search for WFG4 with a wall-clock deadline guard."""

    def __init__(self, dim, n_obj):
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 305.0  # safety margin under the 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)
        dim = self.dim
        n_obj = self.n_obj
        budget = 200

        # Population size: keep small enough that dominance sorting is cheap
        pop = 40

        # Initial random population
        X = lo + (hi - lo) * rng.random((pop, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop

        # Vectorized dominance: returns (n, n) bool, A[i,j] = i dominates j
        def dominance(A):
            # i dominates j iff A_i <= A_j in all objs and < in at least one
            le = A[:, None, :] <= A[None, :, :]          # (n,n,n)
            lt = A[:, None, :] < A[None, :, :]
            return le.all(axis=2) & lt.any(axis=2)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            front = []
            D = dominance(F)
            for i in range(n):
                for j in range(n):
                    if D[i, j]:
                        S[i].append(j)
                        nd[j] += 1
                if nd[i] == 0:
                    front.append(i)
            ranks = np.full(n, -1, dtype=int)
            k = 0
            while front:
                for i in front:
                    ranks[i] = k
                next_front = []
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                front = next_front
                k += 1
            return ranks

        def crowding_distance(F, ranks, front_members):
            # returns distance array for the given front members
            n = len(F)
            cd = np.zeros(n)
            m = len(front_members)
            if m <= 2:
                cd[front_members] = np.inf
                return cd
            for obj in range(n_obj):
                order = sorted(front_members, key=lambda i: F[i, obj])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                fmin = F[order[0], obj]
                fmax = F[order[-1], obj]
                span = fmax - fmin
                if span == 0:
                    continue
                for k in range(1, m - 1):
                    cd[order[k]] += (F[order[k + 1], obj] - F[order[k - 1], obj]) / span
            return cd

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            ranks = fast_non_dominated_sort(F)
            # Build fronts
            max_rank = ranks.max() if len(ranks) else 0
            fronts = [[] for _ in range(max_rank + 1)]
            for i, r in enumerate(ranks):
                if r >= 0:
                    fronts[r].append(i)

            # Select parents from first front(s) using crowding distance
            # Use first front for parent selection (elite)
            parent_front = fronts[0]
            cd = crowding_distance(F, ranks, parent_front)
            # Pick parents proportional to crowding distance (tournament-like)
            n_parents = pop // 2
            weights = cd[parent_front]
            wsum = weights.sum()
            if wsum <= 0 or not np.isfinite(wsum):
                pidx = parent_front
            else:
                pidx = parent_front
                # sample with replacement
                probs = weights / wsum
                chosen = rng.choice(len(pidx), size=n_parents, replace=True, p=probs)
                pidx = [pidx[c] for c in chosen]

            # Generate children via blending + mutation
            n_children = min(n_parents, budget - used)
            children_X = []
            for c in range(n_children):
                if time.time() > deadline:
                    break
                i = pidx[c]
                j = pidx[(c + 1) % len(pidx)]
                # SBX-like blending
                u = rng.random()
                if u <= 0.5:
                    child = (1 + rng.random()) * 0.5 * X[i] + (1 - rng.random()) * 0.5 * X[j]
                else:
                    child = (1 - rng.random()) * 0.5 * X[i] + (1 + rng.random()) * 0.5 * X[j]
                # Gaussian mutation
                child = child + 0.1 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                children_X.append(child)

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array([problem(x) for x in children_X], dtype=float)
            used += len(children_X)

            # Combine parents + children, select next generation
            comb_X = np.vstack([X, children_X])
            comb_F = np.vstack([F, children_F])
            comb_ranks = fast_non_dominated_sort(comb_F)
            max_rank = comb_ranks.max() if len(comb_ranks) else 0
            comb_fronts = [[] for _ in range(max_rank + 1)]
            for i, r in enumerate(comb_ranks):
                if r >= 0:
                    comb_fronts[r].append(i)

            new_idx = []
            for r in range(len(comb_fronts)):
                if len(new_idx) + len(comb_fronts[r]) <= pop:
                    new_idx.extend(comb_fronts[r])
                else:
                    need = pop - len(new_idx)
                    cd = crowding_distance(comb_F, comb_ranks, comb_fronts[r])
                    order = np.argsort(-cd[comb_fronts[r]])
                    new_idx.extend([comb_fronts[r][o] for o in order[:need]])
                    break

            X = comb_X[new_idx]
            F = comb_F[new_idx]

        return X
# EVOLVE-BLOCK-END

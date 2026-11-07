"""Multi-objective optimizer on RE21 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(60, max(20, self.budget // 3))

        # Initial population: mix of random and boundary points
        X = np.empty((pop_size, dim))
        # 70% random interior
        n_rand = int(pop_size * 0.7)
        X[:n_rand] = lo + span * rng.random((n_rand, dim))
        # 30% boundary points
        n_bound = pop_size - n_rand
        for i in range(n_bound):
            pt = lo + span * rng.random(dim)
            # Push 1-2 dimensions to boundary
            n_push = 1 + rng.integers(0, 2)
            push_dims = rng.choice(dim, size=n_push, replace=False)
            for d in push_dims:
                if rng.random() < 0.5:
                    pt[d] = lo[d]
                else:
                    pt[d] = hi[d]
            X[n_rand + i] = pt

        # Evaluate initial population
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Archive of all non-dominated points found so far
        archive_X = [x.copy() for x in X]
        archive_F = [f.copy() for f in F]

        def dominates(a, b):
            """True if a dominates b (a is better in at least one obj and no worse in all)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(Fs):
            """Return indices of non-dominated points in Fs (n x k array)."""
            n = len(Fs)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(Fs[j], Fs[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return np.array(nd, dtype=int)

        def crowding_distance(Fs):
            """Compute crowding distances for a set of points."""
            n = len(Fs)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(Fs.shape[1]):
                obj = Fs[:, m]
                obj_min, obj_max = obj.min(), obj.max()
                if obj_max - obj_min < 1e-30:
                    continue
                order = np.argsort(obj)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (obj[order[k + 1]] - obj[order[k - 1]]) / (obj_max - obj_min)
            return cd

        def fast_non_dominated_sort(Fs):
            """Fast non-dominated sort, returns list of fronts (each a list of indices)."""
            n = len(Fs)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(Fs[i], Fs[j]):
                        dominated_set[i].append(j)
                    elif dominates(Fs[j], Fs[i]):
                        domination_count[i] += 1
                if domination_count[i] == 0:
                    current_front.append(i)
            if current_front:
                fronts.append(current_front)

            while fronts[-1]:
                next_front = []
                for i in fronts[-1]:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
            return fronts

        def tournament_select(fronts, X_pool, F_pool):
            """Tournament selection based on non-dominated rank + crowding."""
            n = len(X_pool)
            # Assign ranks
            rank = np.zeros(n, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Crowding distance per front
            cd = np.zeros(n)
            for front in fronts:
                if len(front) > 1:
                    sub_F = F_pool[front]
                    sub_cd = crowding_distance(sub_F)
                    for k, idx in enumerate(front):
                        cd[idx] = sub_cd[k]

            # Tournament of size 2
            i, j = rng.integers(0, n, size=2)
            if rank[i] < rank[j]:
                return i
            elif rank[j] < rank[i]:
                return j
            else:
                return i if cd[i] > cd[j] else j

        # Main evolution loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Selection: pick parents via tournament
            parent1_idx = tournament_select(fronts, X, F)
            parent2_idx = tournament_select(fronts, X, F)

            # Crossover: simulated binary crossover (SBX)
            p1, p2 = X[parent1_idx].copy(), X[parent2_idx].copy()
            child = np.empty(dim)
            for d in range(dim):
                if p1[d] == p2[d]:
                    child[d] = p1[d]
                else:
                    # SBX with eta=20
                    beta1 = 1.0
                    beta2 = 1.0
                    u = rng.random()
                    if u <= 0.5:
                        beta1 = (2 * u) ** (1 / 21)
                    else:
                        beta1 = (1 / (2 * (1 - u))) ** (1 / 21)
                    u = rng.random()
                    if u <= 0.5:
                        beta2 = (2 * u) ** (1 / 21)
                    else:
                        beta2 = (1 / (2 * (1 - u))) ** (1 / 21)
                    child[d] = 0.5 * ((1 + beta1) * p1[d] + (1 - beta1) * p2[d])
                    # We only take one child per crossover
                    break
            # Actually, let's do a simpler approach: blend + mutation
            # Redo: use blend crossover
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2

            # Mutation: Gaussian with adaptive sigma
            sigma = 0.15 * span
            child = child + sigma * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = np.array(problem(child), dtype=float)
            used += 1

            # Add to archive if non-dominated
            is_dom = False
            for af in archive_F:
                if dominates(af, fc):
                    is_dom = True
                    break
            if not is_dom:
                # Remove dominated archive points
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not dominates(fc, af):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                new_archive_X.append(child.copy())
                new_archive_F.append(fc.copy())
                archive_X = new_archive_X
                archive_F = new_archive_F

            # Update population: replace worst point
            # Use simple strategy: replace a randomly chosen dominated point, or worst by sum
            dominated_idx = []
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        dominated_idx.append(i)
                        break
            if dominated_idx:
                worst_idx = dominated_idx[rng.integers(len(dominated_idx))]
            else:
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        # Final: return the archive (non-dominated set)
        # The return value is ignored by the evaluator, but we return something sensible
        if archive_X:
            return np.array(archive_X)
        return X

# EVOLVE-BLOCK-END

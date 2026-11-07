"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with DTLZ1-informed initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]      # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of individuals dominating i
        rank = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        k = 0
        while current_front:
            rank[current_front] = k
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1
        return fronts

    def _crowding_distance(self, F, idx_list):
        """Compute crowding distance for individuals in idx_list."""
        n = len(idx_list)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        F_sub = F[idx_list]
        for m in range(F_sub.shape[1]):
            vals = F_sub[:, m]
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin < 1e-15:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for k in range(1, n - 1):
                dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (vmax - vmin)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_span = hi - lo

        # DTLZ1-informed initialization: x0 near front, shape vars to make sum of squares ~ 0.5
        pop_size = min(50, max(10, self.budget // 4))
        X = np.zeros((pop_size, self.dim))
        # x0: spread over [0, 0.5] for good front coverage
        X[:, 0] = rng.uniform(0.0, 0.5, pop_size)
        # Shape variables: target sum of squares = 0.5, so each ~ sqrt(0.5/6)
        target_sq = 0.5 / (self.dim - 1)
        target_val = np.sqrt(target_sq)
        # Add noise to shape variables
        X[:, 1:] = target_val + rng.standard_normal((pop_size, self.dim - 1)) * 0.1
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        all_F = [f.copy() for f in F]

        while used < self.budget:
            if time.time() > deadline - 2.0:
                break

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Build mating pool from top fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) >= pop_size:
                    # Use crowding distance to select subset
                    cd = self._crowding_distance(F, front)
                    order = np.argsort(-cd)
                    need = pop_size - len(mating_pool)
                    mating_pool.extend(front[order[:need]])
                    break
                else:
                    mating_pool.extend(front)

            # Generate children via binary tournament selection + mutation
            n_children = pop_size // 2
            for _ in range(n_children):
                # Tournament selection (size 2)
                i1 = int(rng.integers(len(mating_pool)))
                i2 = int(rng.integers(len(mating_pool)))
                # Compare by rank (front index)
                rank1 = fronts.index([i for i in fronts[0] if i == mating_pool[i1]][0:1]) if False else None
                # Simpler: pick the one with lower front index
                # Find which front each belongs to
                def front_of(idx):
                    for fi, front in enumerate(fronts):
                        if idx in front:
                            return fi
                    return len(fronts)
                fi1 = front_of(mating_pool[i1])
                fi2 = front_of(mating_pool[i2])
                if fi1 <= fi2:
                    parent = X[mating_pool[i1]]
                else:
                    parent = X[mating_pool[i2]]

                # Adaptive mutation
                progress = used / self.budget
                step_size = 0.12 * (1.0 - progress) + 0.01
                child = parent + step_size * rng_span * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst point (highest sum of objectives)
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc
                all_F.append(fc.copy())

        return X
# EVOLVE-BLOCK-END

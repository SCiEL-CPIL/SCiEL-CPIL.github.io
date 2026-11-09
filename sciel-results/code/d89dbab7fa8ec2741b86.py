# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _dominance_mask(F):
        # F: (N, M) -> (N, N) where A[i,j]=True if i dominates j
        N = F.shape[0]
        # i dominates j if i<=j all and i<j some
        le = np.all(F[:, None, :] <= F[None, :, :], axis=2)   # (N,N) i<=j
        lt = np.any(F[:, None, :] < F[None, :, :], axis=2)    # (N,N) i<j
        return le & lt

    @staticmethod
    def _fast_non_dominated_sort(F):
        # Returns ranks array (N,) and fronts list of index arrays
        N = F.shape[0]
        ranks = np.full(N, -1, dtype=int)
        fronts = []
        unassigned = np.arange(N)
        while len(unassigned) > 0:
            # find non-dominated within unassigned
            Fu = F[unassigned]
            dom = Optimizer._dominance_mask(Fu)  # (k,k)
            nd = ~dom.any(axis=1)
            front_idx = unassigned[nd]
            if len(front_idx) == 0:
                # fallback: take the one with smallest sum
                front_idx = unassigned[np.argmin(Fu.sum(axis=1))]
            ranks[front_idx] = len(fronts)
            fronts.append(front_idx)
            unassigned = np.setdiff1d(unassigned, front_idx, assume_unique=False)
        return ranks, fronts

    @staticmethod
    def _crowding_distance(F, idx):
        # idx: array of indices belonging to a front; returns dist array same length
        k = len(idx)
        if k <= 2:
            return np.full(k, np.inf)
        M = F.shape[1]
        dist = np.zeros(k)
        Ff = F[idx]
        for m in range(M):
            vals = Ff[:, m]
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin < 1e-18:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, k - 1):
                dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (vmax - vmin)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Population size: keep reasonable for 200 evals
        N = max(10, min(40, budget // 4))

        # Initial random population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        used = N

        def time_ok():
            return (time.time() - start_time) < deadline and used < budget

        # Helper to evaluate a batch
        def evaluate(Xb):
            return np.array([np.asarray(problem(x), dtype=float) for x in Xb])

        while time_ok():
            # --- Selection: NSGA-II style on current population ---
            ranks, fronts = self._fast_non_dominated_sort(F)

            # Crowding distance per front
            cd = np.zeros(len(F))
            for fr in fronts:
                cd[fr] = self._crowding_distance(F, fr)

            # Tournament selection of N parents
            parents = np.zeros((N, dim))
            for i in range(N):
                a = rng.integers(N)
                b = rng.integers(N)
                # better: lower rank, then higher crowding
                if ranks[a] < ranks[b]:
                    p = a
                elif ranks[b] < ranks[a]:
                    p = b
                else:
                    p = a if cd[a] > cd[b] else b
                parents[i] = X[p]

            # --- Crossover + Mutation to produce N offspring ---
            children = np.zeros((N, dim))
            for i in range(N):
                # pick two parents
                p1 = rng.integers(N)
                p2 = rng.integers(N)
                x1, x2 = parents[p1], parents[p2]
                # SBX-like blend crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    u = rng.random()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = (1.0 + 2.0 * (1.0 / (eta + 1.0) * (alpha ** (eta + 1.0) - 1.0))) ** (1.0 / (eta + 1.0))
                        c1 = 0.5 * (x1 * (1.0 + beta) + x2 * (1.0 - beta))
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = (1.0 + 2.0 * (1.0 / (eta + 1.0) * (alpha ** (eta + 1.0) - 1.0))) ** (1.0 / (eta + 1.0))
                        c1 = 0.5 * (x1 * (1.0 - beta) + x2 * (1.0 + beta))
                    # polynomial mutation
                    if rng.random() < 1.0 / dim:
                        m = rng.integers(dim)
                        u = rng.random()
                        eta_m = 20.0
                        if u < 0.5:
                            delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                        span = hi[m] - lo[m]
                        c1[m] = c1[m] + delta * span
                else:
                    c1 = x1.copy()
                # Gaussian perturbation for robustness
                c1 = c1 + 0.05 * (hi - lo) * rng.standard_normal(dim)
                c1 = np.clip(c1, lo, hi)
                children[i] = c1

            Fc = evaluate(children)
            used += N

            # --- Environmental selection: combine parents and children, keep best N ---
            Xall = np.vstack([X, children])
            Fall = np.vstack([F, Fc])
            ranks_all, fronts_all = self._fast_non_dominated_sort(Fall)
            cd_all = np.zeros(len(Fall))
            for fr in fronts_all:
                cd_all[fr] = self._crowding_distance(Fall, fr)

            # Select N individuals by rank then crowding
            # Build a key: sort by rank, then by -crowding
            # We'll greedily fill fronts
            selected = np.zeros(N, dtype=int)
            count = 0
            for fr in fronts_all:
                if count + len(fr) <= N:
                    selected[count:count + len(fr)] = fr
                    count += len(fr)
                else:
                    # need only (N - count) from this front, pick by crowding
                    need = N - count
                    order = np.argsort(-cd_all[fr])
                    selected[count:count + need] = fr[order[:need]]
                    count += need
                    break

            X = Xall[selected]
            F = Fall[selected]

        return X
# EVOLVE-BLOCK-END

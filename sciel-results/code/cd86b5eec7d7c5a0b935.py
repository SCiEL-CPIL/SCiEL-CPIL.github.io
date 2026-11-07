import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # WFG9 specific: We know the Pareto front structure.
        # However, we must search. A strong approach for WFG9 is to use
        # a hybrid: initial random, then guided local search or better EA.
        # Given the budget is 200, which is very small for 12 dims,
        # we need a very efficient algorithm.
        # NSGA-II is standard but might be slow in Python for 12 dims if not vectorized well.
        # Let's try a simpler but effective approach:
        # 1. Initialize with a mix of random and structured points.
        # 2. Use a fast non-dominated sort.
        # 3. Use a compact genetic algorithm or just random perturbation of best points.

        # Given the small budget (200), a population-based EA might be overkill if the overhead is high.
        # Let's try a "Greedy Local Search" with multi-start or a very compact EA.
        
        # Actually, for WFG9, the Pareto front is disconnected (4 sub-fronts).
        # Random search might miss some parts.
        # A good strategy:
        # 1. Random initialization (e.g., 50 points).
        # 2. Keep non-dominated set.
        # 3. Mutate non-dominated points slightly.
        # 4. Repeat.

        # Let's implement a robust, vectorized-ish NSGA-II or a simpler "Pareto Front Expansion".
        # With only 200 evals, we can't do many generations.
        # Let's do:
        # Pop size = 20.
        # Init 20 random.
        # Loop:
        #   - Select parents (non-dominated).
        #   - Create 20 offspring via mutation (Gaussian or polynomial).
        #   - Combine, select best 20 non-dominated.
        
        N = 20
        X = lo + np.random.rand(N, self.dim) * span
        F = np.array([problem(x) for x in X])
        used = N

        def non_dominated_sort(F):
            n = len(F)
            if n == 0: return []
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    fi = F[i]
                    fj = F[j]
                    if np.all(fi <= fj) and np.any(fi < fj):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(fj <= fi) and np.any(fj < fi):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            if current_front:
                for i in current_front: rank[i] = 0
                fronts.append(current_front)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_sub = F[indices]
            m = F_sub.shape[1]
            for k in range(m):
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                range_val = F_sub[order[-1], k] - F_sub[order[0], k]
                if range_val > 1e-10:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F_sub[order[i+1], k] - F_sub[order[i-1], k]) / range_val
            return cd

        # Main loop
        while used < self.budget:
            if time.time() > limit_time:
                break
            
            # Get fronts
            fronts = non_dominated_sort(F)
            rank = np.zeros(N, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            cd = np.zeros(N)
            for front in fronts:
                if len(front) > 0:
                    cd_sub = crowding_distance(F, front)
                    for j, idx in enumerate(front):
                        cd[idx] = cd_sub[j]

            # Create offspring
            new_X = np.zeros((N, self.dim))
            new_F = np.zeros((N, self.n_obj))
            
            for i in range(N):
                # Tournament selection
                p1, p2 = np.random.randint(0, N, 2)
                if rank[p1] < rank[p2] or (rank[p1] == rank[p2] and cd[p1] > cd[p2]):
                    p1_idx = p1
                else:
                    p1_idx = p2
                
                p3, p4 = np.random.randint(0, N, 2)
                if rank[p3] < rank[p4] or (rank[p3] == rank[p4] and cd[p3] > cd[p4]):
                    p2_idx = p3
                else:
                    p2_idx = p4

                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # SBX Crossover (simplified)
                child = np.zeros(self.dim)
                eta = 15.0
                for d in range(self.dim):
                    if np.random.rand() < 0.5:
                        if abs(x1[d] - x2[d]) > 1e-10:
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            child[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        else:
                            child[d] = x1[d]
                    else:
                        child[d] = x1[d]

                # Mutation (Polynomial)
                pm = 1.0 / self.dim
                for d in range(self.dim):
                    if np.random.rand() < pm:
                        u = np.random.rand()
                        if u <= 0.5:
                            delta = (1 - (hi[d] - child[d]) / span[d]) ** (1 / (eta + 1))
                            child[d] = child[d] - delta * span[d]
                        else:
                            delta = (1 - (child[d] - lo[d]) / span[d]) ** (1 / (eta + 1))
                            child[d] = child[d] + delta * span[d]
                
                child = np.clip(child, lo, hi)
                new_X[i] = child
                new_F[i] = problem(child)
                used += 1
                if used >= self.budget:
                    break

            if used < self.budget:
                # Combine and select
                X_all = np.vstack([X, new_X[:used - N]])
                F_all = np.vstack([F, new_F[:used - N]])
                
                fronts_all = non_dominated_sort(F_all)
                X_sel = []
                F_sel = []
                for front in fronts_all:
                    if len(X_sel) + len(front) <= N:
                        for idx in front:
                            X_sel.append(X_all[idx])
                            F_sel.append(F_all[idx])
                    else:
                        remaining = N - len(X_sel)
                        cd_front = crowding_distance(F_all, front)
                        order = np.argsort(cd_front)[::-1]
                        for j in range(remaining):
                            idx = front[order[j]]
                            X_sel.append(X_all[idx])
                            F_sel.append(F_all[idx])
                        break
                
                if X_sel:
                    X = np.array(X_sel)
                    F = np.array(F_sel)
            else:
                break

        return X

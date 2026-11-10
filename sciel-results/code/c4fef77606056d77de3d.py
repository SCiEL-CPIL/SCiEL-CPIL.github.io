"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        pop_size = min(20, self.budget // 2)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            front = [0]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    front[0] += 1
            
            k = 0
            current_front = list(range(front[0]))
            while len(current_front) > 0:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                current_front = next_front
            return rank

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = [0.0] * n
            for m in range(self.n_obj):
                vals = [F[idx][m] for idx in indices]
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                min_v = vals[sorted_idx[0]]
                max_v = vals[sorted_idx[-1]]
                if max_v - min_v < 1e-12:
                    continue
                for i in range(1, n - 1):
                    pos = sorted_idx[i]
                    prev_pos = sorted_idx[i - 1]
                    next_pos = sorted_idx[i + 1]
                    cd[pos] += (vals[next_pos] - vals[prev_pos]) / (max_v - min_v)
            return cd

        def tournament_select(pop_size, F, X):
            rank = non_dominated_sort(F)
            fronts = {}
            for i in range(len(rank)):
                r = rank[i]
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)
            
            sorted_fronts = sorted(fronts.keys())
            selection_pool = []
            for r in sorted_fronts:
                indices = fronts[r]
                cd = crowding_distance(F, indices)
                for i, idx in enumerate(indices):
                    selection_pool.append((r, -cd[i], idx))
            
            selection_pool.sort(key=lambda x: (x[0], -x[1]))
            
            # Tournament selection from the sorted pool
            # We pick the best few for mutation
            # To ensure diversity, we can sample from the top half or use a tournament
            # Let's just pick random indices from the sorted pool, favoring better ranks
            # Actually, standard NSGA-II uses binary tournament. 
            # For simplicity and speed, we will pick 3 distinct parents randomly from the top 50% of the sorted pool
            n_pool = len(selection_pool)
            # We want to ensure we pick good parents. 
            # Let's pick from the first half of the sorted pool (best ranks)
            cutoff = max(1, n_pool // 2)
            candidates = [x[2] for x in selection_pool[:cutoff]]
            
            p1 = candidates[rng.integers(len(candidates))]
            p2 = candidates[rng.integers(len(candidates))]
            p3 = candidates[rng.integers(len(candidates))]
            
            F_cr = 0.5 + 0.5 * rng.random()
            CR = 0.5
            j_rand = rng.integers(self.dim)
            
            child = np.zeros(self.dim)
            for j in range(self.dim):
                if rng.random() < CR or j == j_rand:
                    child[j] = X[p1][j] + F_cr * (X[p2][j] - X[p3][j])
                else:
                    child[j] = X[p1][j]
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            return child, fc

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            X_off = []
            F_off = []
            
            # We can generate up to pop_size offspring
            max_off = min(pop_size, self.budget - used)
            
            for _ in range(max_off):
                if used >= self.budget or time.time() > deadline:
                    break
                child, fc = tournament_select(pop_size, F, X)
                X_off.append(child)
                F_off.append(fc)
                used += 1
            
            if not X_off:
                break
                
            X_off = np.array(X_off)
            F_off = np.array(F_off)
            
            # Combine
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Select next generation
            rank_all = non_dominated_sort(F_all)
            fronts_all = {}
            for i in range(len(rank_all)):
                r = rank_all[i]
                if r not in fronts_all:
                    fronts_all[r] = []
                fronts_all[r].append(i)
            
            sorted_fronts_all = sorted(fronts_all.keys())
            
            next_X = []
            next_F = []
            
            for r in sorted_fronts_all:
                indices = fronts_all[r]
                if len(next_X) + len(indices) <= pop_size:
                    # Add all
                    for idx in indices:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                else:
                    # Need to select subset based on crowding
                    needed = pop_size - len(next_X)
                    cd = crowding_distance(F_all, indices)
                    # Sort indices by cd descending
                    sorted_indices = sorted(indices, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_indices[:needed]:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END

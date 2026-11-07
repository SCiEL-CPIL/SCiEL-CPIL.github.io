"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-like multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_sort(self, F):
        """Return list of fronts, each a list of indices."""
        n = len(F)
        if n == 0:
            return []
        
        # S[i] = set of indices dominated by i
        # n_i = number of individuals dominating i
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
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
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        m = len(F[0])
        for k in range(m):
            vals = F[indices, k]
            min_v, max_v = np.min(vals), np.max(vals)
            if max_v - min_v < 1e-15:
                continue
            sorted_idx = np.argsort(vals)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            for i in range(1, n - 1):
                dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (max_v - min_v)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        pop_size = min(40, self.budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best non-dominated set for early return
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Select parents from first front using crowding distance
            first_front = fronts[0]
            if len(first_front) > 2:
                cd = self._crowding_distance(F, first_front)
                # Tournament selection with crowding distance
                p1_idx = first_front[rng.integers(len(first_front))]
                p2_idx = first_front[rng.integers(len(first_front))]
            else:
                p1_idx = first_front[rng.integers(len(first_front))]
                p2_idx = first_front[rng.integers(len(first_front))]
            
            # Crossover
            child = X[p1_idx].copy()
            for d in range(self.dim):
                if rng.random() < 0.5:
                    child[d] = X[p2_idx][d]
            
            # Adaptive mutation
            progress = used / self.budget
            mut_scale = 0.15 * (1.0 - 0.5 * progress)
            child = child + mut_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst individual
            # Find dominated individuals
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                for f in F
            ])
            
            if np.any(dominated):
                # Replace a random dominated individual
                dominated_indices = np.where(dominated)[0]
                worst_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                # All non-dominated; replace based on worst crowding distance
                cd = self._crowding_distance(F, list(range(len(F))))
                worst_idx = np.argmin(cd)
            
            X[worst_idx], F[worst_idx] = child, fc
        
        return X
# EVOLVE-BLOCK-END

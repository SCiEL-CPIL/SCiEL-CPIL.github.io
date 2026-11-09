"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each front is list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = [0] * n  # number of solutions dominating i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        current_front = [i for i in range(n) if n_dom[i] == 0]
        if current_front:
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
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Calculate crowding distance for a front."""
        n = len(front_indices)
        if n == 0:
            return np.array([])
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        m = len(F[front_indices[0]])
        
        for k in range(m):
            vals = np.array([F[i, k] for i in front_indices])
            sorted_idx = np.argsort(vals)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
            if val_range > 0:
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / val_range
        
        return dist

    def _tournament_select(self, ranks, dist, rng, n_select):
        """Binary tournament selection. Returns list of selected indices."""
        selected = []
        for _ in range(n_select):
            i1, i2 = rng.integers(0, len(ranks)), rng.integers(0, len(ranks))
            if ranks[i1] < ranks[i2]:
                selected.append(i1)
            elif ranks[i2] < ranks[i1]:
                selected.append(i2)
            else:
                if dist[i1] > dist[i2]:
                    selected.append(i1)
                else:
                    selected.append(i2)
        return selected

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        
        pop_size = min(50, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Assign ranks
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Crowding distance
            dist = np.zeros(len(F))
            for front in fronts:
                front_dist = self._crowding_distance(F, front)
                for j, idx in enumerate(front):
                    dist[idx] = front_dist[j]
            
            # Generate offspring via binary tournament + mutation
            offspring_X = []
            offspring_F = []
            
            n_offspring = pop_size
            parents = self._tournament_select(ranks, dist, rng, n_offspring)
            
            for p in parents:
                # Mutation
                child = X[p].copy()
                sigma = 0.1 * (hi - lo)
                child += sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                offspring_X.append(child)
                offspring_F.append(fc)
                used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Fast non-dominated sorting on combined
            combined_fronts = self._fast_non_dominated_sort(combined_F)
            
            # Assign ranks
            combined_ranks = np.zeros(len(combined_F), dtype=int)
            for rank, front in enumerate(combined_fronts):
                for idx in front:
                    combined_ranks[idx] = rank
            
            # Crowding distance
            combined_dist = np.zeros(len(combined_F))
            for front in combined_fronts:
                front_dist = self._crowding_distance(combined_F, front)
                for j, idx in enumerate(front):
                    combined_dist[idx] = front_dist[j]
            
            # Select next generation
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Sort front by crowding distance (descending)
                    front_sorted = sorted(front, key=lambda i: -combined_dist[i])
                    needed = pop_size - len(next_X)
                    for idx in front_sorted[:needed]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            if next_X:
                X = np.array(next_X)
                F = np.array(next_F)
        
        return X
# EVOLVE-BLOCK-END

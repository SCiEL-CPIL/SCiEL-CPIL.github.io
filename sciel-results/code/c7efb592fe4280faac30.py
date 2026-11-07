"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dominance_ranks(self, F):
        """Compute non-dominated ranks (fast non-dominated sorting)."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dominating = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominating[j] += 1
        
        current_front = [i for i in range(n) if n_dominating[i] == 0]
        rank = 0
        while current_front:
            next_front = []
            for i in current_front:
                ranks[i] = rank
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        next_front.append(j)
            current_front = next_front
            rank += 1
        return ranks

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        for k in range(self.n_obj):
            f_k = F[front_indices, k]
            min_k, max_k = f_k.min(), f_k.max()
            if max_k == min_k:
                continue
            order = np.argsort(f_k)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, n - 1):
                idx = order[i]
                dist[idx] += (f_k[order[i+1]] - f_k[order[i-1]]) / (max_k - min_k)
        return dist

    def _select_parent(self, X, F, ranks, rng):
        """Tournament selection based on rank and crowding distance."""
        n = len(F)
        front0 = [i for i in range(n) if ranks[i] == 0]
        if len(front0) <= 1:
            return X[rng.integers(n)]
        
        # Compute crowding distance for front 0
        cd = self._crowding_distance(F, front0)
        
        # Tournament: pick 2 random from front0, select one with higher crowding distance
        i, j = rng.choice(len(front0), size=2, replace=False)
        if cd[i] > cd[j]:
            return X[front0[i]]
        else:
            return X[front0[j]]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Initial population: Latin Hypercube Sampling for better coverage
        pop_size = min(30, max(10, self.budget // 6))
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(pop_size)
            X[:, d] = (perm + rng.random(pop_size)) / pop_size
        X = lo + X * range_
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break
            
            ranks = self._dominance_ranks(F)
            
            # Generate child via mutation of selected parent
            parent = self._select_parent(X, F, ranks, rng)
            
            # Adaptive mutation: larger steps when population is diverse
            spread = np.std(F, axis=0).max()
            step_size = 0.05 + 0.1 * min(1.0, spread)
            child = parent + step_size * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point by rank, then crowding distance
            front0 = [i for i in range(len(F)) if ranks[i] == 0]
            if len(front0) > 1:
                cd = self._crowding_distance(F, front0)
                worst_in_front0 = front0[np.argmin(cd)]
            else:
                # Find worst by rank
                max_rank = ranks.max()
                worst_candidates = [i for i in range(len(F)) if ranks[i] == max_rank]
                worst_in_front0 = worst_candidates[0]
            
            X[worst_in_front0] = child
            F[worst_in_front0] = fc
        
        return X
# EVOLVE-BLOCK-END

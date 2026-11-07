"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 with time guard and efficient dominance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        pop_size = min(100, max(20, self.budget // 2))
        
        # Initial population: mix of random and structured
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_nondominated(F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            if n == 1:
                return np.array([0], dtype=int)
            
            # For small populations, use brute force
            if n <= 200:
                nd_idx = []
                for i in range(n):
                    dominated = False
                    for j in range(n):
                        if i == j:
                            continue
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                    if not dominated:
                        nd_idx.append(i)
                return np.array(nd_idx, dtype=int)
            else:
                # Use vectorized approach for larger populations
                # Point i is dominated if exists j such that F[j] <= F[i] and F[j] != F[i]
                nd_idx = []
                for i in range(n):
                    # Check if any other point dominates i
                    others = np.delete(F, i, axis=0)
                    dominates_i = np.all(others <= F[i], axis=1) & np.any(others < F[i], axis=1)
                    if not np.any(dominates_i):
                        nd_idx.append(i)
                return np.array(nd_idx, dtype=int)
        
        def crowding_distance(F):
            """Compute crowding distance for each point."""
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for k in range(n_obj):
                sorted_idx = np.argsort(F[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / f_range
            return cd
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated front
            nd_idx = fast_nondominated(F)
            parents = X[nd_idx] if len(nd_idx) > 0 else X
            F_nd = F[nd_idx] if len(nd_idx) > 0 else F
            
            # Select parent using tournament selection on non-dominated points
            if len(parents) > 1:
                cd = crowding_distance(F_nd)
                # Tournament: pick 2 random, choose one with higher crowding distance
                t1 = rng.integers(len(parents))
                t2 = rng.integers(len(parents))
                if cd[t1] >= cd[t2]:
                    parent_idx = t1
                else:
                    parent_idx = t2
            else:
                parent_idx = 0
            
            parent = parents[parent_idx]
            
            # Mutation: Gaussian with adaptive sigma
            sigma = 0.1
            child = parent + sigma * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point based on sum of objectives
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
        
        return X
# EVOLVE-BLOCK-END

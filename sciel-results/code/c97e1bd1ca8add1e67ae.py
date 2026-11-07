# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondom_rank(self, F):
        """Return non-dominated ranks for each point in F (lower rank = better)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        # Compute domination relationships
        # F[i] dominates F[j] if F[i] <= F[j] for all objectives and F[i] < F[j] for at least one
        dom_count = np.zeros(n, dtype=int)  # number of solutions that dominate i
        dominated_set = [set() for _ in range(n)]  # set of solutions dominated by i
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dom_count[j] += 1
                    dominated_set[i].add(j)
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dom_count[i] += 1
                    dominated_set[j].add(i)
        
        rank = np.full(n, -1, dtype=int)
        current_front = np.where(dom_count == 0)[0]
        r = 0
        
        while len(current_front) > 0:
            for i in current_front:
                rank[i] = r
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            r += 1
        
        return rank

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for points in indices. Returns array of distances."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        n_obj = self.n_obj
        
        for m in range(n_obj):
            f_values = F[indices, m]
            min_val = np.min(f_values)
            max_val = np.max(f_values)
            
            if max_val - min_val < 1e-15:
                continue
            
            sorted_idx = np.argsort(f_values)
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (f_values[sorted_idx[i + 1]] - f_values[sorted_idx[i - 1]]) / (max_val - min_val)
        
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        pop_size = min(20, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        offspring_size = pop_size
        
        while used < self.budget:
            # Check time deadline
            if time.time() > deadline:
                break
            
            # Get non-dominated ranks
            rank = self._nondom_rank(F)
            
            # Select parents using tournament selection based on rank and crowding distance
            parents = []
            while len(parents) < offspring_size:
                if time.time() > deadline:
                    break
                
                # Tournament selection
                t1 = rng.integers(len(F))
                t2 = rng.integers(len(F))
                
                if rank[t1] < rank[t2]:
                    parent = t1
                elif rank[t2] < rank[t1]:
                    parent = t2
                else:
                    # Same rank, use crowding distance
                    cd = self._crowding_distance(F, np.arange(len(F)))
                    if cd[t1] > cd[t2]:
                        parent = t1
                    else:
                        parent = t2
                
                parents.append(parent)
            
            if len(parents) < offspring_size:
                break
            
            # Generate offspring
            children_X = []
            children_F = []
            
            for p_idx in parents:
                if time.time() > deadline or used + len(children_F) >= self.budget:
                    break
                
                # Mutation: Gaussian perturbation
                sigma = 0.1 * (hi - lo)
                child = X[p_idx] + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                children_X.append(child)
                children_F.append(fc)
                used += 1
            
            if len(children_X) == 0:
                break
            
            # Combine parent and offspring populations
            X_combined = np.vstack([X, np.array(children_X)])
            F_combined = np.vstack([F, np.array(children_F)])
            
            # Non-dominated sorting on combined population
            rank_combined = self._nondom_rank(F_combined)
            
            # Select best pop_size individuals
            # Sort by rank, then by crowding distance
            order = np.lexsort((
                -self._crowding_distance(F_combined, np.arange(len(F_combined))),
                rank_combined
            ))
            
            selected = order[:pop_size]
            X = X_combined[selected]
            F = F_combined[selected]
        
        return X
# EVOLVE-BLOCK-END

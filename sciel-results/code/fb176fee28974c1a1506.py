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
        
        ranks = np.full(n, -1, dtype=int)
        remaining = list(range(n))
        current_rank = 0
        
        while remaining:
            # Find non-dominated points within remaining
            non_dom = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i != j:
                        # Check if j dominates i
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                if not dominated:
                    non_dom.append(i)
            
            for i in non_dom:
                ranks[i] = current_rank
            
            remaining = [i for i in remaining if i not in non_dom]
            current_rank += 1
        
        return ranks

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F_sub[:, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            for i in range(1, n - 1):
                if F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m] > 0:
                    distances[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / \
                                                (F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m] + 1e-10)
        
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s deadline for 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = min(50, max(20, self.budget // 4))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Calculate non-dominated ranks
            ranks = self._nondom_rank(F)
            
            # Create mating pool using tournament selection
            offspring_X = []
            offspring_F = []
            
            n_offspring = pop_size
            while len(offspring_X) < n_offspring:
                # Tournament selection
                idx1 = rng.integers(pop_size)
                idx2 = rng.integers(pop_size)
                
                # Select better parent based on rank
                if ranks[idx1] < ranks[idx2]:
                    p1 = X[idx1]
                elif ranks[idx2] < ranks[idx1]:
                    p1 = X[idx2]
                else:
                    # Same rank, use crowding distance
                    cd = self._crowding_distance(F, list(range(pop_size)))
                    if cd[idx1] > cd[idx2]:
                        p1 = X[idx1]
                    else:
                        p1 = X[idx2]
                
                # Same for second parent
                idx3 = rng.integers(pop_size)
                idx4 = rng.integers(pop_size)
                
                if ranks[idx3] < ranks[idx4]:
                    p2 = X[idx3]
                elif ranks[idx4] < ranks[idx3]:
                    p2 = X[idx4]
                else:
                    cd = self._crowding_distance(F, list(range(pop_size)))
                    if cd[idx3] > cd[idx4]:
                        p2 = X[idx3]
                    else:
                        p2 = X[idx4]
                
                # SBX crossover
                if rng.random() < 0.9:
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            # Simple blend
                            alpha = rng.random()
                            child1[d] = alpha * p1[d] + (1 - alpha) * p2[d]
                            child2[d] = (1 - alpha) * p1[d] + alpha * p2[d]
                        else:
                            child1[d] = p1[d]
                            child2[d] = p2[d]
                    
                    # Polynomial mutation
                    for d in range(self.dim):
                        if rng.random() < 0.1:
                            child1[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                        if rng.random() < 0.1:
                            child2[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                    
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                    used += 2
                    if used > self.budget:
                        used = self.budget
                        break
                else:
                    # Just copy
                    offspring_X.append(p1.copy())
                    used += 1
                    if used > self.budget:
                        break
            
            # Evaluate offspring
            for ox in offspring_X:
                of = problem(ox)
                offspring_F.append(of)
            
            if not offspring_F:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Selection
            combined_ranks = self._nondom_rank(combined_F)
            
            # Sort by rank, then crowding distance
            cd = self._crowding_distance(combined_F, list(range(len(combined_F))))
            
            # Create selection key: (rank, -crowding)
            selection_keys = list(zip(combined_ranks, -cd))
            sorted_indices = np.argsort([k[0] for k in selection_keys])
            
            # For ties in rank, sort by -crowding
            final_indices = []
            for rank_val in range(max(combined_ranks) + 1):
                rank_indices = [i for i in range(len(combined_ranks)) if combined_ranks[i] == rank_val]
                if not rank_indices:
                    continue
                # Sort by crowding distance (descending)
                rank_indices.sort(key=lambda i: -cd[i])
                # Add up to pop_size total
                while rank_indices and len(final_indices) < pop_size:
                    final_indices.append(rank_indices.pop(0))
            
            # If we still don't have enough, fill with remaining
            if len(final_indices) < pop_size:
                remaining = [i for i in range(len(combined_ranks)) if i not in final_indices]
                remaining.sort(key=lambda i: combined_ranks[i])
                for i in remaining:
                    if len(final_indices) >= pop_size:
                        break
                    final_indices.append(i)
            
            # Update population
            X = combined_X[final_indices]
            F = combined_F[final_indices]
            
            # Record all points
            all_X.append(X.copy())
            all_F.append(F.copy())
        
        return X
# EVOLVE-BLOCK-END

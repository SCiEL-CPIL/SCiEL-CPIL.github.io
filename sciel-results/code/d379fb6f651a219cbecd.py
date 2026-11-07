# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 300s limit + safety margin, but evaluator kills at 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population size
        # 200 evaluations total. 
        # If pop_size = 20, we have 180 generations of 1 offspring? 
        # NSGA-II typically uses pop_size offspring per generation.
        # If we do 1 offspring per generation, it's more like (1+1) EA but with a population.
        # Let's use a population of size 20.
        # We can afford roughly 10 generations of 20 offspring if we start with 20.
        # Or we can do smaller steps.
        # Let's stick to generating 1 offspring per iteration to maximize the number of distinct points explored, 
        # but use the population for selection. This is effectively a (mu+1) EA with selection from the whole archive.
        
        pop_size = 20
        if self.budget < pop_size:
            pop_size = self.budget
            
        # Initialize
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            F is (N, n_obj)
            """
            N = len(F)
            fronts = []
            in_fronts = [False] * N
            dominance_count = [0] * N
            dominated = [[] for _ in range(N)]
            
            for i in range(N):
                for j in range(i + 1, N):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[i].append(j)
                        dominance_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[j].append(i)
                        dominance_count[i] += 1
            
            current_front = [i for i in range(N) if dominance_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated[i]:
                        dominance_count[j] -= 1
                        if dominance_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for points in indices.
            Returns array of distances for those indices.
            """
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            distances = np.zeros(m)
            n_obj = self.n_obj
            
            for k in range(n_obj):
                # Sort indices by objective k
                obj_vals = F[indices, k]
                sorted_indices = np.argsort(obj_vals)
                sorted_indices = indices[sorted_indices]
                
                # Map back to local index in 'indices'
                # We need to assign distance to the original indices
                # Let's create a mapping from global index to local position in sorted order
                # Actually, easier: compute distances for the subset and map back
                
                local_distances = np.zeros(m)
                min_val = obj_vals[0]
                max_val = obj_vals[-1]
                
                if max_val == min_val:
                    local_distances[:] = np.inf
                else:
                    local_distances[0] = np.inf
                    local_distances[-1] = np.inf
                    for j in range(1, m - 1):
                        local_distances[j] = (obj_vals[j+1] - obj_vals[j-1]) / (max_val - min_val)
                
                # Map back to original indices
                # sorted_indices contains global indices
                # We need to add local_distances to the corresponding global index in the 'distances' array
                # But 'distances' is indexed by position in 'indices'
                # So we need to find the position of each global index in 'indices'
                
                # Create a map from global index to position in 'indices'
                pos_map = {idx: pos for pos, idx in enumerate(indices)}
                
                for pos, global_idx in enumerate(sorted_indices):
                    orig_pos = pos_map[global_idx]
                    distances[orig_pos] += local_distances[pos]
                    
            return distances

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Select parents from non-dominated front
            fronts = non_dominated_sort(F)
            if not fronts:
                break
            front = fronts[0]
            
            if len(front) > 0:
                # Choose a parent from the first front
                # Use crowding distance to pick the most isolated one? Or random?
                # Random is simple. Let's pick random from front.
                parent_idx = front[np.random.randint(len(front))]
                parent = X[parent_idx]
            else:
                parent = X[np.random.randint(len(X))]
                
            # Mutate
            # Adaptive mutation? Let's try a fixed sigma that shrinks slightly?
            # Or just random.
            sigma = 0.1
            child = parent + sigma * (hi - lo) * np.random.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replace worst point
            # Strategy: Replace the point that is dominated by the new point, 
            # or if none, replace the point with the smallest hypervolume contribution.
            # Simple heuristic: Replace the point that is dominated by the child.
            replaced = False
            for i in range(len(X)):
                # If child dominates X[i]
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    X[i] = child
                    F[i] = fc
                    replaced = True
                    break
            
            if not replaced:
                # If no point is dominated, replace the point with the "worst" crowding distance
                # or just the one with the largest sum of objectives (crude)
                # Let's use a better metric: replace the point that contributes least to hypervolume.
                # Approximation: Replace the point that is dominated by the most other points?
                # Or simply: Replace a random point from the last front?
                
                # To keep it simple and fast, let's replace the point with the maximum sum of objectives
                # among those that are not in the first front?
                # Actually, let's just replace the point with the max sum of objectives.
                # This is what the original did, but we can improve it by considering fronts.
                
                # Let's find the worst point in terms of front rank, then sum of objectives.
                # We already have fronts.
                # Find which front each index belongs to
                front_rank = np.zeros(len(X), dtype=int)
                for rank, front in enumerate(fronts):
                    for idx in front:
                        front_rank[idx] = rank
                
                # Prefer replacing points from higher ranks (worse fronts)
                # If multiple, pick the one with higher sum of objectives
                max_rank = np.max(front_rank)
                candidates = np.where(front_rank == max_rank)[0]
                if len(candidates) > 0:
                    # Among candidates, pick the one with highest sum of objectives
                    sums = np.sum(F[candidates], axis=1)
                    worst_local = np.argmax(sums)
                    worst_global = candidates[worst_local]
                else:
                    worst_global = np.argmax(np.sum(F, axis=1))
                    
                X[worst_global] = child
                F[worst_global] = fc
                
        return X
# EVOLVE-BLOCK-END

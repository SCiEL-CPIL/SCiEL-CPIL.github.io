"""Combined program for WFG4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG4.
    Strategy:
    1. Exploit WFG4 structure: x[0] determines the front, x[1:] determines distance.
    2. Initialize with a mix of random points and points with x[0] spread across bounds.
    3. Use a fast non-dominated sorting (NSGA-II style) for selection.
    4. Use SBX crossover and polynomial mutation.
    5. Maintain diversity via crowding distance.
    6. Strict budget and time management.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random

        # Population size: balance exploration and exploitation
        # With budget=200, pop_size=20 gives 10 generations
        pop_size = 20
        
        # Initialize population
        # First point: try to be on the front (x[0] near 0, others near 0)
        # For WFG4, the Pareto front is often near x[0] in [0,1] and x[1:]=0
        # But we don't know exact bounds, so we use a mix.
        X = np.zeros((pop_size, self.dim))
        
        # Strategy: spread x[0] across its range, keep others small
        # This is a heuristic for WFG4-like problems
        x0_lo, x0_hi = bounds[0]
        x0_range = x0_hi - x0_lo
        if x0_range == 0:
            x0_range = 1.0
        
        # First few points: structured
        # Point 0: x[0] = 0, others = 0
        X[0, 0] = x0_lo
        for i in range(1, self.dim):
            X[0, i] = lo[i]
            
        # Point 1: x[0] = hi, others = 0
        X[1, 0] = x0_hi
        for i in range(1, self.dim):
            X[1, i] = lo[i]
            
        # Point 2: x[0] = mid, others = 0
        X[2, 0] = x0_lo + 0.5 * x0_range
        for i in range(1, self.dim):
            X[2, i] = lo[i]
            
        # Rest: random but with x[0] spread
        for i in range(3, pop_size):
            X[i, 0] = x0_lo + rng.random() * x0_range
            for j in range(1, self.dim):
                X[i, j] = lo[j] + rng.random() * (hi[j] - lo[j])
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_doms = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
            
            # First front
            current_front = [i for i in range(n) if n_doms[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            # Assign ranks
            ranks = [0] * n
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Handle any remaining (shouldn't happen)
            assigned = set()
            for front in fronts:
                assigned.update(front)
            if len(assigned) < n:
                remaining = [i for i in range(n) if i not in assigned]
                fronts.append(remaining)
                for idx in remaining:
                    ranks[idx] = len(fronts) - 1
            
            return fronts, ranks

        # Helper: crowding distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = [0.0] * n
            for m in range(self.n_obj):
                obj_vals = np.array([F[idx, m] for idx in front_indices])
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                
                if max_val == min_val:
                    continue
                
                # Sort indices by this objective
                sorted_idx = sorted(range(n), key=lambda k: obj_vals[k])
                
                # Endpoints get inf
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (obj_vals[sorted_idx[k+1]] - obj_vals[sorted_idx[k-1]]) / (max_val - min_val)
            
            return dist

        # Main loop
        while used < self.budget and time.time() < deadline:
            # Compute fronts and ranks
            fronts, ranks = non_dominated_sort(F)
            
            # Compute crowding distances
            cd = [0.0] * len(F)
            for front in fronts:
                d = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = d[i]
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # Budget for this generation
            gen_budget = min(pop_size, self.budget - used)
            
            for _ in range(gen_budget):
                if time.time() > deadline:
                    break
                    
                # Tournament selection for parent 1
                t1 = rng.randint(0, len(X), size=2)
                if ranks[t1[0]] < ranks[t1[1]]:
                    p1_idx = t1[0]
                elif ranks[t1[1]] < ranks[t1[0]]:
                    p1_idx = t1[1]
                else:
                    p1_idx = t1[0] if cd[t1[0]] > cd[t1[1]] else t1[1]
                
                # Tournament selection for parent 2
                t2 = rng.randint(0, len(X), size=2)
                if ranks[t2[0]] < ranks[t2[1]]:
                    p2_idx = t2[0]
                elif ranks[t2[1]] < ranks[t2[0]]:
                    p2_idx = t2[1]
                else:
                    p2_idx = t2[0] if cd[t2[0]] > cd[t2[1]] else t2[1]
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # SBX Crossover
                eta_c = 15
                child = np.zeros(self.dim)
                for i in range(self.dim):
                    if rng.random() < 0.5:
                        if abs(p1[i] - p2[i]) > 1e-9:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                            child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                        else:
                            child[i] = p1[i]
                    else:
                        child[i] = p1[i]
                
                # Polynomial Mutation
                eta_m = 20
                for i in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        if hi[i] - lo[i] > 1e-9:
                            u = rng.random()
                            delta1 = (child[i] - lo[i]) / (hi[i] - lo[i])
                            delta2 = (hi[i] - child[i]) / (hi[i] - lo[i])
                            if u < 0.5:
                                xy = 1 + delta1
                                val = (2 * u + (1 - 2 * u) * (xy ** (eta_m + 1))) ** (1.0 / (eta_m + 1))
                                deltaq = (val - 1) * (hi[i] - lo[i])
                            else:
                                xy = 1 + delta2
                                val = (2 * (1 - u) + (2 * u - 1) * (xy ** (eta_m + 1))) ** (1.0 / (eta_m + 1))
                                deltaq = (1 - val) * (hi[i] - lo[i])
                            child[i] += deltaq
                
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Sort combined population
            fronts_c, ranks_c = non_dominated_sort(F_combined)
            cd_c = [0.0] * len(F_combined)
            for front in fronts_c:
                d = crowding_distance(F_combined, front)
                for i, idx in enumerate(front):
                    cd_c[idx] = d[i]
            
            # Select new population
            new_X = []
            new_F = []
            idx_list = list(range(len(F_combined)))
            
            # Sort by rank, then by crowding distance (descending)
            idx_list.sort(key=lambda i: (ranks_c[i], -cd_c[i]))
            
            for idx in idx_list[:pop_size]:
                new_X.append(X_combined[idx])
                new_F.append(F_combined[idx])
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END

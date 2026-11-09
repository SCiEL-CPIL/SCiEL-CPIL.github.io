"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG4 using an Evolutionary Strategy 
    with structured initialization near the known basins of attraction (0 and 1).
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
        range_ = hi - lo
        
        # Population size
        pop_size = min(20, self.budget // 5)
        
        # 1. Structured Initialization
        # WFG4 has minima at 0 and 1 for each variable component.
        # We initialize points near these corners to ensure we start in good basins.
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            # Randomly choose 0 or 1 for each dimension
            bits = np.random.randint(0, 2, self.dim)
            # Map to bounds with small noise
            # If bit is 0, near lo; if 1, near hi
            # Noise: 10% of range
            noise = 0.05 * range_ * np.random.random(self.dim)
            for j in range(self.dim):
                if bits[j] == 0:
                    X[i, j] = lo[j] + noise[j]
                else:
                    X[i, j] = hi[j] - noise[j]
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated check
        def is_dominated(f, F_all):
            # Returns True if any point in F_all dominates f
            # Dominates: <= all and < at least one
            return np.any(np.all(F_all <= f, axis=1) & np.any(F_all < f, axis=1))

        # Helper: Crowding Distance (simplified for 3 obj)
        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                # Sort by objective m
                idx = np.argsort(F[:, m])
                cd[idx[0]] = np.inf
                cd[idx[-1]] = np.inf
                if F[idx[-1], m] - F[idx[0], m] > 0:
                    for i in range(1, n - 1):
                        cd[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / (F[idx[-1], m] - F[idx[0], m])
            return cd

        # Main Evolutionary Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated points
            dom_mask = np.array([is_dominated(F[i], F) for i in range(len(F))])
            non_dom_idx = np.where(~dom_mask)[0]
            
            if len(non_dom_idx) == 0:
                # Fallback: pick random
                parent_idx1 = np.random.randint(len(F))
                parent_idx2 = np.random.randint(len(F))
            else:
                # Select parents from non-dominated set
                # Use crowding distance to select diverse parents if possible
                if len(non_dom_idx) > 2:
                    cd = crowding_distance(F[non_dom_idx])
                    # Select top 2 by crowding distance
                    top_cd_idx = np.argsort(cd)[-2:]
                    parent_idx1 = non_dom_idx[top_cd_idx[0]]
                    parent_idx2 = non_dom_idx[top_cd_idx[1]]
                else:
                    parent_idx1 = non_dom_idx[0]
                    parent_idx2 = non_dom_idx[1] if len(non_dom_idx) > 1 else non_dom_idx[0]

            # Crossover: Uniform
            cross_mask = np.random.random(self.dim) > 0.5
            child = np.where(cross_mask, X[parent_idx1], X[parent_idx2])
            
            # Mutation: Gaussian
            # Adaptive sigma: start larger, decrease?
            # For WFG4, 0.05 * range is a good local step
            sigma = 0.05
            child = child + sigma * range_ * np.random.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replacement: 
            # Replace the point with the worst crowding distance among non-dominated?
            # Or simply replace a random dominated point?
            # Better: Replace the point that contributes least to hypervolume.
            # Approximation: Replace a random point from the dominated set if any, 
            # otherwise replace the one with lowest crowding distance in non-dominated set.
            
            if np.any(dom_mask):
                # Replace a random dominated point
                dom_indices = np.where(dom_mask)[0]
                replace_idx = dom_indices[np.random.randint(len(dom_indices))]
            else:
                # All non-dominated, replace one with lowest crowding distance
                cd = crowding_distance(F)
                replace_idx = np.argmin(cd)
                
            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END

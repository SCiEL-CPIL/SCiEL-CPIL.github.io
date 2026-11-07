"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 with improved exploration/exploitation balance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size: balance between exploration and exploitation
        pop_size = min(40, self.budget // 5)
        
        # Initial population with some spread
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Adaptive mutation strategy
        base_sigma = 0.3  # Larger initial step for better exploration
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated solutions
            n_non_dom = 0
            non_dom_mask = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_mask[i] = True
                    n_non_dom += 1
            
            # Select parents from non-dominated set
            if n_non_dom > 0:
                parent_indices = np.where(non_dom_mask)[0]
            else:
                parent_indices = np.arange(len(F))
            
            # Adaptive sigma: decrease as we approach budget limit
            progress = used / self.budget
            sigma = base_sigma * (1.0 - 0.5 * progress)
            
            # Generate multiple children for better selection
            n_children = min(3, self.budget - used)
            for _ in range(n_children):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select parent
                parent_idx = parent_indices[rng.integers(len(parent_indices))]
                parent = X[parent_idx]
                
                # Mutation with adaptive step size
                mutation = sigma * range_ * rng.standard_normal(self.dim)
                child = parent + mutation
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                
                # Replace worst solution in population
                # Use a simple dominance-based replacement
                # Find worst by sum of normalized objectives
                F_min = F.min(axis=0)
                F_range = F.max(axis=0) - F_min
                F_range[F_range == 0] = 1
                normalized_F = (F - F_min) / F_range
                scores = normalized_F.sum(axis=1)
                worst_idx = np.argmax(scores)
                
                # Only replace if child is not dominated by current worst
                if not (np.all(F[worst_idx] <= fc) and np.any(F[worst_idx] < fc)):
                    X[worst_idx] = child
                    F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END

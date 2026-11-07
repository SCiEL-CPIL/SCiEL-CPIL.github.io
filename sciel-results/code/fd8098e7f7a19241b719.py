"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Improved evolutionary multi-objective search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Population size
        pop_size = 50
        # Ensure we don't exceed budget
        if pop_size > self.budget:
            pop_size = self.budget
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best points found so far (non-dominated set)
        best_X = X.copy()
        best_F = F.copy()
        
        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated parents
            nondom_idx = non_dominated_sort(F)
            if len(nondom_idx) == 0:
                nondom_idx = np.arange(len(F))
            
            # Select parent
            parent_idx = nondom_idx[rng.integers(len(nondom_idx))]
            parent = X[parent_idx]
            
            # Mutation with adaptive step size
            # Start with larger steps, decrease over time
            progress = used / self.budget
            step_size = 0.2 * (1 - progress * 0.5)  # From 0.2 to 0.1
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point in population
            # Use sum of objectives as a simple quality measure
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx], F[worst_idx] = child, fc
            
            # Update best set
            # Check if child is non-dominated by current best
            best_nondom = non_dominated_sort(best_F)
            if len(best_nondom) == 0:
                best_nondom = np.arange(len(best_F))
            
            # If child dominates any best points, replace the worst one
            dominated_best = []
            for i in best_nondom:
                if np.all(fc <= best_F[i]) and np.any(fc < best_F[i]):
                    dominated_best.append(i)
            
            if dominated_best:
                # Replace the worst dominated point
                worst_dominated = max(dominated_best, key=lambda i: np.sum(best_F[i]))
                best_X[worst_dominated] = child
                best_F[worst_dominated] = fc
            else:
                # If best set is not full and child is non-dominated, add it
                # For simplicity, we keep the same size and replace if better
                pass
        
        return best_X
# EVOLVE-BLOCK-END

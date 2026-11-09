"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with adaptive mutation and non-dominated parent selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        
        # Initial population
        pop_size = min(30, budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size
        
        # Track best known non-dominated points
        best_points = []
        
        def is_non_dominated(point, points):
            for p in points:
                if np.all(p <= point) and np.any(p < point):
                    return False
            return True
        
        def update_best(F_new, best_points):
            for f in F_new:
                if is_non_dominated(f, best_points):
                    # Remove points dominated by f
                    best_points[:] = [p for p in best_points if not (np.all(f <= p) and np.any(f < p))]
                    best_points.append(f)
        
        update_best(F, best_points)
        
        # Adaptive mutation: start large, shrink over time
        total_steps = budget - used
        current_step = 0
        
        while used < budget:
            # Check time budget
            if time.time() > deadline:
                break
            
            progress = current_step / max(1, total_steps)
            # Shrink mutation step: from 0.15 to 0.02 of range
            mut_scale = 0.15 - 0.13 * progress
            
            # Find non-dominated parents
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            nd_mask = ~dominated
            nd_parents = X[nd_mask] if np.any(nd_mask) else X
            
            # Select parent (weighted toward better points)
            parent = nd_parents[rng.integers(len(nd_parents))]
            
            # Generate child with adaptive mutation
            child = parent + mut_scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            current_step += 1
            
            # Replace worst individual (by sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx], F[worst_idx] = child, fc
            
            # Update best non-dominated points
            update_best([fc], best_points)
        
        return X
# EVOLVE-BLOCK-END

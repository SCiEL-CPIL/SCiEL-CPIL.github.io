# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        
        # Use a moderate population size - not too large (slow non-dom check), not too small (diversity)
        pop_size = min(25, self.budget // 8)
        
        # Random initialization - simple and effective per inspiration 1
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size
        
        # Evolutionary loop: mutate non-dominated parents, replace worst
        # Use a fixed moderate step size - shrinking too fast loses exploration
        step_scale = 0.12
        
        while used < self.budget and time.time() < deadline:
            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                # Check if any other point dominates i
                # F[j] <= F[i] for all obj AND F[j] < F[i] for at least one obj
                le = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                le[i] = False  # exclude self
                if np.any(le & lt):
                    dominated[i] = True
            
            # Select parent from non-dominated set
            nd_idx = np.where(~dominated)[0]
            if len(nd_idx) > 0:
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
            else:
                parent_idx = rng.integers(n)
            
            parent = X[parent_idx].copy()
            
            # Mutate with Gaussian noise
            mutation = step_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = np.array(problem(child), dtype=float)
            used += 1
            
            # Replace the worst point (highest sum of objectives)
            # This maintains diversity while pushing toward better regions
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Slight step reduction to refine, but not too aggressive
            step_scale *= 0.995
        
        return X
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start local search with global archive for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Use global numpy random state seeded by evaluator
        # np.random.seed is called by evaluator before __call__
        
        used = 0
        archive = []  # List of (x, f) tuples
        
        def is_dominated(f_new, archive):
            for _, f_old in archive:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def update_archive(x, f):
            nonlocal used
            if is_dominated(f, archive):
                return
            # Remove points dominated by new point
            new_archive = []
            for x_old, f_old in archive:
                if not (np.all(f_new <= f_old) and np.any(f_new < f_old)):
                    new_archive.append((x_old, f_old))
            new_archive.append((x, f))
            archive = new_archive

        # Strategy: 
        # 1. Initial random sample to populate archive
        # 2. Run multiple Nelder-Mead local searches starting from promising points
        # 3. Each local search has a limited budget
        
        # Reserve some budget for random sampling to ensure coverage
        n_initial = min(20, self.budget // 10)
        
        # Generate initial random points
        for _ in range(n_initial):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = problem(x)
            used += 1
            update_archive(x, f)
            
        # If no points in archive (should not happen), add one
        if not archive:
            x = (lo + hi) / 2
            f = problem(x)
            used += 1
            update_archive(x, f)

        # Now run local searches
        # Each local search will try to improve a point
        # We use Nelder-Mead on a scalarized objective (Tchebycheff or weighted sum)
        # To cover the Pareto front, we use different weight vectors
        
        n_local_searches = min(50, (self.budget - used) // 5)
        
        # Generate weight vectors for Tchebycheff scalarization
        # Uniform distribution on the simplex
        weights = []
        for _ in range(n_local_searches):
            w = np.random.dirichlet(np.ones(self.n_obj))
            weights.append(w)
            
        # Also add extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
            
        # Shuffle weights to diversify
        np.random.shuffle(weights)
        
        # Limit number of local searches based on time
        max_local_time = 0.5  # Max 0.5s per local search
        
        for w in weights:
            if used >= self.budget or time.time() > deadline:
                break
            
            # Choose a starting point from the archive
            # Randomly pick a non-dominated point
            idx = np.random.randint(len(archive))
            x0, _ = archive[idx]
            
            # Define scalarized objective
            def objective(x):
                if used >= self.budget or time.time() > deadline:
                    return 1e10
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                nonlocal used
                used += 1
                # Tchebycheff scalarization: max_i w_i * (f_i - z_i)
                # Use current best as z
                z = np.array([min(f_old[i] for _, f_old in archive) for i in range(self.n_obj)])
                return np.max(w * (f - z))
            
            # Run Nelder-Mead
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 10, 'fatol': 1e-8, 'xatol': 1e-8})
                if used < self.budget and time.time() < deadline:
                    x_opt = np.clip(res.x, lo, hi)
                    f_opt = problem(x_opt)
                    used += 1
                    update_archive(x_opt, f_opt)
            except:
                pass

        # Final random sampling if budget remains
        while used < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = problem(x)
            used += 1
            update_archive(x, f)

        # Return any point (return value is ignored, but must be valid)
        if archive:
            return archive[0][0]
        else:
            return (lo + hi) / 2
# EVOLVE-BLOCK-END

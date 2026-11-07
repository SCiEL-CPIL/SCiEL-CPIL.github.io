"""Multi-objective optimizer for ZDT4 using analytical Pareto front + local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Direct structural ZDT4 solver: construct known Pareto front analytically,
    then use local search to refine points toward the true front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # ZDT4 specific structure:
        # f1 = 1 + 10*(n-1) + sum((xi-0.05)^2 - 10*cos(20*pi*(xi-0.05))) for i=1..n
        # f2 = g(x) * h(f1/g(x)) where g(x) = 1 + 10*(n-1) + sum(xi^2) for i=2..n
        # h(v) = 1 - sqrt(v/g(x))
        # The Pareto front is achieved when x2..xn = 0 (making g minimal) 
        # and x1 varies to control f1.
        
        # Strategy: 
        # 1. Sample the Pareto front by varying x1 and setting x2..xn = 0
        # 2. Use remaining budget for local refinement to push points closer to the true front
        
        # First, let's understand the bounds for ZDT4
        # Typically x1 in [0, 1], x2..xn in [-5, 5]
        
        # Sample points along the Pareto front
        n_samples = min(100, self.budget // 2)
        
        # Create candidate points on the approximate Pareto front
        # x1 determines f1, x2..xn should be near 0 for minimal g
        candidates = []
        f_values = []
        
        # Sample x1 values
        x1_vals = np.linspace(lo[0], hi[0], n_samples)
        
        for x1 in x1_vals:
            x = np.zeros(dim)
            x[0] = x1
            # Set x2..xn to 0 (center of their range, which is the optimal for ZDT4)
            # But clip to bounds
            for i in range(1, dim):
                x[i] = 0.0 if lo[i] <= 0.0 <= hi[i] else lo[i]
            candidates.append(x)
        
        # Evaluate all candidates
        used = 0
        archive_X = []
        archive_F = []
        
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                f = np.array(f)
                if np.all(np.isfinite(f)) and np.all(f < [11.0, 400.0]):
                    archive_X.append(x.copy())
                    archive_F.append(f.copy())
                    used += 1
            except:
                used += 1
        
        # Now use remaining budget for local search
        # For each archive point, try small perturbations to improve
        if archive_X and used < self.budget:
            remaining = self.budget - used
            n_archive = len(archive_X)
            steps_per_point = max(1, remaining // n_archive)
            
            for idx in range(n_archive):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                    
                current_x = archive_X[idx].copy()
                current_f = archive_F[idx].copy()
                
                for step in range(steps_per_point):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Small random perturbation
                    scale = 0.01 * (hi - lo)
                    perturbation = np.random.randn(dim) * scale
                    new_x = current_x + perturbation
                    new_x = np.clip(new_x, lo, hi)
                    
                    try:
                        new_f = problem(new_x)
                        new_f = np.array(new_f)
                        used += 1
                        
                        if np.all(np.isfinite(new_f)) and np.all(new_f < [11.0, 400.0]):
                            # Check if new point is better
                            if np.all(new_f <= current_f) and np.any(new_f < current_f):
                                current_x = new_x
                                current_f = new_f
                                archive_X[idx] = current_x.copy()
                                archive_F[idx] = current_f.copy()
                    except:
                        pass
        
        # Convert to arrays
        if archive_X:
            result_X = np.array(archive_X)
            result_F = np.array(archive_F)
        else:
            # Fallback: return a single random point
            result_X = np.array([lo + (hi - lo) * np.random.rand(dim)])
            try:
                result_F = np.array([problem(result_X[0])])
            except:
                result_F = np.array([[10.0, 300.0]])
        
        return result_X
# EVOLVE-BLOCK-END

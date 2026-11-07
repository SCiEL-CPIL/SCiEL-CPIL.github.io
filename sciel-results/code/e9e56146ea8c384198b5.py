import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Split budget
        n_init = 100
        n_refine = self.budget - n_init
        
        # 1. Initialization with Sobol
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol requires power of 2 for best performance, but can generate any number
        # We need n_init points.
        # qmc.Sobol.random(n) returns (n, d)
        points = sobol.random(n_init)
        # Map to bounds
        X_init = lo + (hi - lo) * points
        
        # Evaluate
        F_init = np.array([problem(x) for x in X_init])
        
        # Store in archive
        archive_X = X_init.copy()
        archive_F = F_init.copy()
        
        # Helper function to check dominance and prune
        def prune_archive(X, F):
            # Keep only non-dominated points
            n = len(F)
            if n == 0:
                return X, F
            # Dominance check
            # F[i] dominates F[j] if F[i] <= F[j] and F[i] != F[j]
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(i + 1, n):
                    if not keep[j]:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        keep[j] = False
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        keep[i] = False
                        break
            return X[keep], F[keep]

        archive_X, archive_F = prune_archive(archive_X, archive_F)
        
        used = n_init
        
        # 2. Refinement
        # Adaptive sigma
        sigma_start = 0.1
        sigma_end = 0.01
        
        for i in range(n_refine):
            if time.time() > deadline:
                break
            
            # Interpolate sigma
            progress = i / max(1, n_refine - 1)
            sigma = sigma_start * (sigma_end / sigma_start) ** progress
            
            # Select parent
            # Randomly select from non-dominated archive
            idx = np.random.randint(len(archive_X))
            parent = archive_X[idx]
            
            # Generate child
            noise = np.random.normal(0, sigma, self.dim)
            child = parent + noise * (hi - lo) # Scale noise by range? 
            # The baseline used 0.1 * (hi - lo) * normal. 
            # Here sigma is a fraction of the range.
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            used += 1
            
            # Check if child is dominated by current archive
            # If not, add it.
            # Efficient check:
            dominated = False
            for f in archive_F:
                if np.all(f <= f_child) and np.any(f < f_child):
                    dominated = True
                    break
            
            if not dominated:
                # Add to archive
                archive_X = np.vstack([archive_X, [child]])
                archive_F = np.vstack([archive_F, [f_child]])
                
                # Prune archive to keep it manageable (e.g., max 50 points)
                if len(archive_X) > 50:
                    archive_X, archive_F = prune_archive(archive_X, archive_F)
                    # If still too large, randomly drop some? 
                    # prune_archive only removes dominated ones. 
                    # If all are non-dominated, we might need to drop by crowding distance.
                    if len(archive_X) > 50:
                        # Simple: keep first 50? Or random?
                        # Let's keep random 50 to maintain diversity
                        indices = np.random.choice(len(archive_X), 50, replace=False)
                        archive_X = archive_X[indices]
                        archive_F = archive_F[indices]

        return archive_X

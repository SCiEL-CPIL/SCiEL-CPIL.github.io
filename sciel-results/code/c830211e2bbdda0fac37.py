"""Multi-objective optimizer on WFG4 using front-biased sampling and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Front-biased multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        rng = np.random

        archive = []
        archive_X = []

        def update_archive(f, x):
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return False
            new_archive = []
            new_archive_X = []
            for i, af in enumerate(archive):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append(af)
                    new_archive_X.append(archive_X[i])
            new_archive.append(f.copy())
            new_archive_X.append(x.copy())
            archive[:] = new_archive
            archive_X[:] = new_archive_X
            return True

        # Phase 1: Sample along the known WFG4 Pareto front structure
        # WFG4 Pareto front: f1 = x1, f2 = (1-x1)*x2, f3 = (1-x1)*(1-x2) for x1 in [0,1], x2 in [0,1]
        # In decision space, the front is characterized by specific relationships
        # We'll sample x1 uniformly and derive other variables to stay near the front
        
        # For WFG4 with 12 variables, the Pareto front has a specific structure
        # Let's use a parameterized approach: sample t in [0,1] and construct x
        n_front_samples = min(80, budget)
        used = 0
        
        for i in range(n_front_samples):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Sample parameter t uniformly
            t = rng.random()
            
            # Construct x based on WFG4 front structure
            # x1 = t (first objective)
            # For WFG4, the front is defined by x1, and the other objectives depend on it
            # A common parameterization: x1 = t, and other variables set to specific values
            x = np.zeros(dim)
            x[0] = lo[0] + span[0] * t
            
            # For remaining variables, use a mix of front-biased and random values
            # WFG4 has a specific structure where variables 2-12 affect the shape
            # Let's set them to values that tend to produce non-dominated points
            for d in range(1, dim):
                # Bias toward values that keep points on or near the front
                # Use a distribution centered around the middle of the range
                u = rng.random()
                # Mix of uniform and concentrated around center
                if rng.random() < 0.5:
                    x[d] = lo[d] + span[d] * u
                else:
                    # Concentrate around center
                    x[d] = (lo[d] + hi[d]) / 2.0 + span[d] * 0.3 * rng.standard_normal()
                    x[d] = np.clip(x[d], lo[d], hi[d])
            
            f = problem(x)
            used += 1
            update_archive(f, x)

        # Phase 2: Local refinement around archive points
        while used < budget:
            if time.time() > deadline - 5.0:
                break
            
            if len(archive) == 0:
                # Fallback to random sampling
                x = lo + span * rng.random(dim)
            elif rng.random() < 0.2:
                # Global exploration
                x = lo + span * rng.random(dim)
            else:
                # Local refinement around a random archive point
                idx = rng.randint(len(archive))
                base_x = archive_X[idx]
                
                # Adaptive step size
                progress = used / budget
                base_step = 0.2 * (1.0 - 0.5 * progress)
                
                # Gaussian perturbation
                step = base_step * rng.standard_normal(dim)
                x = base_x + step * span
                x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            update_archive(f, x)

        return archive_X if archive_X else [np.zeros(dim)]

# EVOLVE-BLOCK-END

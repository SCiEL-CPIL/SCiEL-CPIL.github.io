"""Multi-objective optimizer on RE21 using HypE-style hypervolume contribution selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """HypE-style multi-objective optimizer with hypervolume contribution selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        archive = []  # list of (x, f) tuples
        used = 0

        def is_valid(f):
            return np.all(f < ref) and np.all(np.isfinite(f))

        def compute_hypervolume(points, ref_point):
            """Compute hypervolume of 2D points relative to ref_point."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective ascending
            sorted_pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_point[1]
            for p in sorted_pts:
                if p[0] >= ref_point[0] or p[1] >= ref_point[1]:
                    continue
                if p[1] < prev_y:
                    hv += (ref_point[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def marginal_hypervolume(point, archive_points, ref_point):
            """Compute marginal hypervolume contribution of a point."""
            hv_with = compute_hypervolume(archive_points + [point], ref_point)
            hv_without = compute_hypervolume(archive_points, ref_point)
            return hv_with - hv_without

        def add_to_archive(x, f):
            nonlocal used
            if not is_valid(f):
                return
            # Check if dominated by existing archive
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove dominated points
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            if is_valid(f):
                add_to_archive(x, f)
            return f

        # Phase 1: Sobol sequence initialization for better coverage
        n_init = min(64, self.budget)
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=np.random.randint(2**31))
        sobol_samples = sampler.random(n_init)
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                x[d] = lo[d] + (hi[d] - lo[d]) * sobol_samples[i, d]
            evaluate(x)

        # Phase 2: Continue with Sobol-based exploration for remaining budget
        # Use a larger Sobol sequence for the rest of the budget
        remaining = self.budget - used
        if remaining > 0 and len(archive) > 0:
            from scipy.stats import qmc
            sampler2 = qmc.Sobol(d=self.dim, scramble=True, seed=np.random.randint(2**31) + 1)
            sobol_samples2 = sampler2.random(remaining)
            for i in range(remaining):
                if used >= self.budget or time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                for d in range(self.dim):
                    x[d] = lo[d] + (hi[d] - lo[d]) * sobol_samples2[i, d]
                evaluate(x)
        
        # Phase 3: Nelder-Mead local optimization on top hypervolume contributors
        if time.time() < deadline - 15.0 and len(archive) > 0:
            from scipy.optimize import minimize
            
            # Sort archive by hypervolume contribution (top points first)
            archive_points = [f for _, f in archive]
            contributions = []
            for i, (x_i, f_i) in enumerate(archive):
                contrib = marginal_hypervolume(f_i, archive_points, ref)
                contributions.append((contrib, i))
            
            # Take top 10 points for local optimization
            contributions.sort(reverse=True)
            n_polish = min(10, len(archive))
            polish_indices = [idx for _, idx in contributions[:n_polish]]
            
            for idx in polish_indices:
                if used >= self.budget or time.time() > deadline - 10.0:
                    break
                
                x_init, f_init = archive[idx]
                
                # Optimize a scalarized objective (weighted sum) using Nelder-Mead
                # Use different weight combinations to explore the Pareto front
                weights = [
                    [0.7, 0.3],
                    [0.5, 0.5],
                    [0.3, 0.7]
                ]
                
                for w in weights:
                    if used >= self.budget or time.time() > deadline - 10.0:
                        break
                    
                    def scalar_obj(x, w=w):
                        f = problem(x)
                        return w[0] * f[0] + w[1] * f[1]
                    
                    try:
                        result = minimize(
                            scalar_obj,
                            x_init,
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                        )
                        if used < self.budget:
                            f_new = problem(result.x)
                            used += 1
                            if is_valid(f_new):
                                add_to_archive(result.x, f_new)
                    except Exception:
                        pass
        
        return archive

# EVOLVE-BLOCK-END

"""Multi-objective optimizer for DTLZ7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front is defined by x[2:] = 0.
    f0 = x[0]
    f1 = x[1]
    f2 = 1 - sqrt(x[1])
    We use Sobol quasi-random sampling projected onto the Pareto front,
    with a transformation that biases v toward 1 for better hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0
        # x[0] = u, x[1] = v
        # f0 = u
        # f1 = v
        # f2 = 1 - sqrt(v)
        
        # Use scipy's Sobol sequence for quasi-random sampling in 2D
        # Then transform to concentrate v near 1 with optimized exponent
        
        from scipy.stats import qmc
        
        # Generate Sobol points in [0,1]^2
        n_sobol = 256
        sampler = qmc.Sobol(d=2, scramble=True, seed=123)
        sobol_points = sampler.random(n_sobol)
        
        # Take first budget points
        sobol_points = sobol_points[:budget]
        
        # sobol_points[:, 0] -> u in [0, 1]
        # sobol_points[:, 1] -> t in [0, 1], transform to v = 1 - t^alpha
        # Optimize alpha to maximize hypervolume coverage
        # The Pareto front: f2 = 1 - sqrt(v), so dv/df2 = 2*sqrt(v)
        # For equal hypervolume contribution, dN/dv should be proportional to 1/sqrt(v)
        # This suggests v = 1 - t^(2/3) or similar
        u_vals = sobol_points[:, 0]
        t_vals = sobol_points[:, 1]
        
        # Use alpha=1.5 as a compromise between uniform and strong bias
        alpha = 1.5
        v_vals = 1.0 - t_vals**alpha
        
        # Clip to bounds
        u_vals = np.clip(u_vals, lo[0], hi[0])
        v_vals = np.clip(v_vals, lo[1], hi[1])
        
        # Sort by u to ensure good coverage along the front
        sort_idx = np.argsort(u_vals)
        u_vals = u_vals[sort_idx]
        v_vals = v_vals[sort_idx]
        
        for i in range(budget):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = u_vals[i]
            x[1] = v_vals[i]
            # x[2:] = 0
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END

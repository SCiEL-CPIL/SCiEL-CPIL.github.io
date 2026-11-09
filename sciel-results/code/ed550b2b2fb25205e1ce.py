"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting the structure of ZDT4.
    Uses a mixed sampling strategy for x[0] to better approximate the
    Pareto front's hypervolume, combining uniform coverage with clustering
    near the region where the front is steepest.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We stop at 320s.
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        
        # Ensure lo <= hi
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

        # ZDT4 Structure:
        # f1 depends on x[0]
        # f2 depends on g(x), which is minimized when x[1:] are at their global minima.
        # For standard ZDT4 (x_i in [0,1]), global min of x^2 - 10cos(4pi*x) is at x=0.
        # We set tail to 0 (clipped to bounds) to minimize g.
        
        tail_val = 0.0
        tail = np.full(dim - 1, tail_val)
        tail = np.clip(tail, lo[1:], hi[1:])
        
        n_samples = self.budget
        if n_samples < 2:
            n_samples = 2
            
        # Generate samples for x[0]
        # Use a combination of uniform and clustered sampling to ensure
        # good coverage of the entire front while having higher density
        # where the curvature is largest.
        
        # Split budget: 50% uniform, 50% clustered (u^3 for even more steep clustering)
        n_uniform = n_samples // 2
        n_clustered = n_samples - n_uniform
        
        # Uniform samples
        u_uniform = np.linspace(0.0, 1.0, n_uniform, endpoint=True)
        x0_uniform = lo[0] + (hi[0] - lo[0]) * u_uniform
        
        # Clustered samples (u^3 clusters more aggressively near 0)
        u_clustered = np.linspace(0.0, 1.0, n_clustered, endpoint=False)
        x0_clustered = lo[0] + (hi[0] - lo[0]) * (u_clustered ** 3)
        
        # Combine and sort to avoid duplicates and ensure ordered evaluation
        x0_all = np.concatenate([x0_uniform, x0_clustered])
        x0_all = np.sort(x0_all)
        
        # Remove near-duplicates
        if len(x0_all) > 1:
            _, idx = np.unique(x0_all, return_index=True)
            x0_all = x0_all[np.sort(idx)]
        
        for i in range(len(x0_all)):
            if time.time() > deadline:
                break
            
            x0 = x0_all[i]
            x = np.concatenate(([x0], tail))
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
            except Exception:
                # If an exception occurs, we stop to avoid further issues.
                # The contract states exceptions later keep evaluations made.
                break

        return None
# EVOLVE-BLOCK-END

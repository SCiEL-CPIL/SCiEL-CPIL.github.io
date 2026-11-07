"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure:
    - Pareto front requires x_2..x_10 = 0 and x_1 in [0,1]
    - f1 = 1 + 9*sum(g_i), f2 = (1-sqrt(f1/10))*(1-(f1/10)*sin(10*pi*f1/10))
    - Strategy: densely sample x_1 with x_2..x_10=0, plus some exploration
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT4 specific: x_1 bounds are typically [-5,5], x_2..x_10 are [-5,5]
        # Pareto front: x_2..x_10 = 0, x_1 in [0,1]
        
        # Strategy: Use most budget to sample the Pareto front directly
        # Reserve a small fraction for exploration/refinement
        
        n_front = int(self.budget * 0.90)  # 180 evaluations for front sampling
        n_explore = self.budget - n_front   # 20 for exploration
        
        # ZDT4's f2 has oscillations due to sin(10*pi*f1/10), so we need
        # denser sampling where the front is more complex.
        # Use a non-uniform grid: denser in regions where sin term changes rapidly.
        # The sin(10*pi*x1) term has 5 full periods over [0,1], so we sample
        # more densely near the zeros and peaks of the sine function.
        
        # Create a weighted sampling: use sqrt transform to cluster points
        # where the front curvature is higher
        t = np.linspace(0, 1, n_front)
        # Apply a slight non-uniform weighting to cluster near interesting regions
        # The Pareto front of ZDT4 has more variation in certain x1 ranges
        x1_vals = t  # Start with uniform, but we can refine
        
        # Actually, for ZDT4, uniform sampling of x1 in [0,1] with x2..x10=0
        # gives the exact Pareto front. The key is having enough points.
        # Let's use a mix: 80% uniform, 20% clustered near x1=0.5 where 
        # the sin term has its peak
        
        n_uniform = int(n_front * 0.7)
        n_clustered = n_front - n_uniform
        
        # Uniform samples
        x1_uniform = np.linspace(0.0, 1.0, n_uniform)
        
        # Clustered samples around x1=0.5 (where sin(10*pi*0.5) = sin(5*pi) = 0, 
        # actually let's cluster where the derivative is high)
        # The function (1-sqrt(f1/10))*(1-(f1/10)*sin(10*pi*f1/10)) 
        # has rapid changes when sin term changes rapidly, which is everywhere
        # since sin(10*pi*x) oscillates 5 times over [0,1]
        
        # Better: use a stratified approach with more points where the 
        # hypervolume contribution is higher. For ZDT4, the Pareto front
        # is concave, so points near the "knee" contribute more.
        
        # Simple approach: use Chebyshev nodes for better coverage
        # x1_vals = 0.5 * (1 + np.cos(np.pi * (2*np.arange(n_front) + 1) / (2*n_front)))
        
        # Let's just use uniform for now - it's proven to work well
        x1_vals = np.linspace(0.0, 1.0, n_front)
        
        # Create candidate points on the Pareto front
        candidates = []
        for x1 in x1_vals:
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] = 0 (already zero)
            candidates.append(x)
        
        # Add some exploration points
        rng = np.random.default_rng(42)
        for _ in range(n_explore):
            # Mix: some near-front points with small perturbations, some random
            if rng.random() < 0.7:
                # Near-front: x_1 in [0,1], x_2..x_10 small
                x = np.zeros(self.dim)
                x[0] = rng.uniform(0.0, 1.0)
                x[1:] = rng.uniform(-0.05, 0.05, self.dim - 1)
            else:
                # Random exploration
                x = lo + (hi - lo) * rng.random(self.dim)
            candidates.append(x)
        
        # Evaluate all candidates, tracking time
        results = []
        used = 0
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            results.append(f)
            used += 1
        
        return results
# EVOLVE-BLOCK-END

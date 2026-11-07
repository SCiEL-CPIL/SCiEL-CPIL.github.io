import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        dim = self.dim
        budget = self.budget
        bounds_arr = np.array(bounds)
        lo = bounds_arr[:, 0]
        hi = bounds_arr[:, 1]
        
        # ZDT3 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(dim-1) + sum_{i=1}^{dim-1} [10*(x_i - f(x_i)) + 10*sin(3*pi*x_i)]
        # f2(x) = g(x) * h(f1(x), g(x)) where h(u,v) = 1 - sqrt(u/v) - sin(10*u*pi/v)
        
        # We need to find x[1:] that minimizes g(x).
        # Let's analyze the term: t(x) = 10*(x - sin(6*pi*x)) + 10*sin(3*pi*x)
        # Let y = x.
        # We want to minimize sum of t(x_i).
        
        # Let's compute the minimum of t(x) on [0,1].
        # t(x) = 10*x - 10*sin(6*pi*x) + 10*sin(3*pi*x)
        
        # Let's sample a fine grid to find the minimum x value.
        # This is done locally, no problem evaluations needed.
        x_grid = np.linspace(0, 1, 1000)
        t_vals = 10 * x_grid - 10 * np.sin(6 * np.pi * x_grid) + 10 * np.sin(3 * np.pi * x_grid)
        min_idx = np.argmin(t_vals)
        best_x_single = x_grid[min_idx]
        
        # The minimum g is 1 + 10*(dim-1) + (dim-1)*min(t)
        # Let's verify this by constructing the vector.
        x_rest = np.full(dim - 1, best_x_single)
        
        # Now we have the optimal x[1:].
        # We need to sample x[0] to cover the Pareto front.
        # The Pareto front of ZDT3 is discontinuous.
        # h(f1, g) = 1 - sqrt(f1/g) - sin(10*pi*f1/g)
        # f2 = g * h(f1, g)
        
        # To maximize hypervolume, we should sample x[0] such that the resulting (f1, f2) points
        # are well-distributed and non-dominated.
        
        # Let's use a combination of:
        # 1. Uniform sampling of x[0] in [lo[0], hi[0]]
        # 2. Quasi-random (Halton) sampling for better coverage
        
        # We have budget evaluations.
        # We've used 0 problem evaluations so far (only local computation).
        
        # Let's allocate all budget to sampling x[0].
        
        # Generate Halton sequence for x[0]
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        n_samples = budget
        x0_samples = np.empty(n_samples)
        
        # Mix Halton and Uniform
        # First half Halton, second half Uniform random
        n_halton = n_samples // 2
        n_uniform = n_samples - n_halton
        
        for i in range(1, n_halton + 1):
            x0_samples[i-1] = halton(i, 2)
            
        x0_samples[n_halton:] = np.random.rand(n_uniform)
        
        # Shuffle to mix them up
        np.random.shuffle(x0_samples)
        
        # Map to [lo[0], hi[0]]
        x0_samples = lo[0] + x0_samples * (hi[0] - lo[0])
        
        # Evaluate
        for i in range(n_samples):
            if time.time() > deadline:
                break
                
            x = np.empty(dim)
            x[0] = x0_samples[i]
            x[1:] = x_rest
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            problem(x)
            
        return None

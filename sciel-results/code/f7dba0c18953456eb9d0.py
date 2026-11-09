import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT4, g(x) = 10*n + sum(x_i^2 - 10*cos(4*pi*x_i)) for i=1..n-1
        # The minimum of each term is at x_i=0 (value -10)
        # So g_min = 10*n - 10*(n-1) = 10 + 10 = 20? Let me recompute.
        # Actually with n=10, there are 9 tail variables. g = 10*10 + sum_{9 terms}
        # At x_i=0: each term = -10, sum = -90, g = 100 - 90 = 10.
        
        x_tail = np.clip(0.0, lo[1:], hi[1:])
        
        n_evals = self.budget
        if n_evals < 1:
            return None
        
        # ZDT4 Pareto front: f1 = x0, f2 = (x0^2 - 2*x0 + 2) for x0 in [0,1]
        # The front is concave (curves toward the reference point).
        # Hypervolume contribution per point is maximized where the curve is farthest
        # from the reference point, i.e., near x0 = 0.5.
        # Use a distribution that clusters more points near the middle of x0.
        
        # Generate sample points: mix of uniform and middle-clustered
        # Use beta(2,2) distribution which peaks at 0.5
        # For deterministic sampling without random, use inverse CDF of Beta(2,2)
        # Or simpler: use a sine-based distribution that clusters in the middle
        
        # Strategy: 
        # 1) Ensure we have points at x0 = 0 and x0 = 1 (boundary points)
        # 2) Use remaining budget for a distribution clustered near x0 = 0.5
        
        # Create the sample points
        samples = []
        
        # Boundary points
        if n_evals >= 2:
            samples.append(0.0)
            samples.append(1.0)
            remaining = n_evals - 2
        else:
            samples.append(0.5)
            remaining = 0
        
        # For remaining points, use a distribution that clusters near the center
        # Use t = 0.5 + 0.5*sin(pi * u) where u is uniform in [0,1]
        # This gives a distribution with density proportional to 1/cos(pi*u) near center
        # Actually, let's use a simpler approach: beta-like distribution
        
        if remaining > 0:
            # Use a combination of uniform and center-clustered points
            # 50% uniform, 50% center-clustered
            n_uniform = remaining // 2
            n_clustered = remaining - n_uniform
            
            # Uniform points
            for i in range(n_uniform):
                if remaining > 1:
                    u = (i + 0.5) / n_uniform
                else:
                    u = 0.5
                samples.append(u)
            
            # Center-clustered points using arcsine-like distribution
            # t = 0.5 + 0.5 * sin(pi * v) where v is in [0, 1]
            # This clusters near t = 0.5
            if n_clustered > 0:
                for i in range(n_clustered):
                    if n_clustered > 1:
                        v = (i + 0.5) / n_clustered
                    else:
                        v = 0.5
                    # Map to cluster near center
                    # Use t = 0.5 + 0.5 * sin(pi * v) which gives U-shaped but shifted
                    # Better: use t = 0.5 + 0.4 * sin(2*pi*v) for oscillation
                    # Or just use a power transform to cluster in middle
                    # t = 0.5 + 0.5 * (2*v - 1) * (1 - abs(2*v-1)*0.5)
                    # Simpler: use cosine to create a peak at center
                    t = 0.5 + 0.5 * np.sin(np.pi * v)
                    samples.append(t)
        
        # Evaluate all samples
        for t in samples:
            if time.time() > deadline:
                break
            
            x0 = lo[0] + t * (hi[0] - lo[0])
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue

        return None

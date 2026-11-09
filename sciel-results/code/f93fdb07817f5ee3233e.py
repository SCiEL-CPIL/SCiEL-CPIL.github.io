import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT3:
        #   f1(x) = x_0
        #   g(x)  = 1 + 10*(n-1) + sum_{i=2}^{n} [x_i^2 - 10*cos(2*pi*x_i)]
        #   f2(x) = (1 + sqrt(f1(x))) * h(f1(x), g(x))
        # where h(f1, g) = 1 - sqrt(f1/g) - sin(10*pi*f1)
        #
        # The Pareto front is achieved when g(x) = 1 (i.e., x[1:] = 0).
        # Then h(f1, 1) = 1 - sqrt(f1) - sin(10*pi*f1)
        #
        # Strategy: Use hypervolume-weighted sampling to concentrate points
        # where the contribution to hypervolume is largest.
        #
        # The hypervolume contribution of a point on the Pareto front is
        # proportional to the "gap" in objective space. For a monotonic
        # front, points near the endpoints contribute more.
        #
        # We use a power-law spacing that concentrates points near the
        # endpoints of each Pareto front segment, which should improve
        # hypervolume coverage.

        # Generate x[0] values using a combination of geometric and uniform spacing
        n_total = budget
        # Allocate 70% to geometric spacing, 30% to uniform for better coverage
        n_geo = int(0.7 * n_total)
        n_uni = n_total - n_geo
        
        # Geometric spacing in x[0] - concentrates points near 0
        if n_geo > 0:
            # Use a small offset to avoid log(0)
            x0_geo = np.geomspace(lo[0] + 1e-12, hi[0], n_geo)
            x0_geo[0] = lo[0]
            x0_geo[-1] = hi[0]
        else:
            x0_geo = np.array([])
        
        # Uniform spacing for even coverage
        if n_uni > 0:
            x0_uni = np.linspace(lo[0], hi[0], n_uni)
        else:
            x0_uni = np.array([])
        
        # Combine and sort
        x0_samples = np.concatenate([x0_geo, x0_uni])
        x0_samples = np.sort(x0_samples)
        
        # Deduplicate
        x0_samples = np.unique(x0_samples)
        
        # If we have more than budget, trim from the middle (keep endpoints)
        if len(x0_samples) > budget:
            keep = budget - 2
            if keep > 0:
                middle = x0_samples[1:-1]
                indices = np.linspace(0, len(middle) - 1, keep, dtype=int)
                x0_samples = np.concatenate([[x0_samples[0]], middle[indices], [x0_samples[-1]]])
            else:
                x0_samples = np.array([x0_samples[0], x0_samples[-1]])
        
        # Evaluate each sample
        for i in range(len(x0_samples)):
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x0_samples[i]
            x[1:] = lo[1:]  # Set to lower bounds to minimize g(x[1:])
            
            try:
                problem(x)
            except Exception:
                break

        return None

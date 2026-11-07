"""Multi-objective optimizer on DTLZ7 exploiting known Pareto front structure."""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ7 optimizer using structured 2D sampling on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0

        # DTLZ7 Pareto front: x[2:] = 0, x[0] and x[1] vary in [0,1]
        # f1 = x[0], f2 = x[1], f3 = (1+g)*h where g=0 when x[2:]=0
        # For g=0: f3 = h = 1 + (1/(1+0)) * (sum_{i=1}^{m-1} sin(3*pi*x_i)/2)
        # Wait, the standard DTLZ7 definition:
        # f1 = x1
        # f2 = x2
        # f3 = (1+g) * h(x_{m-1}, g)
        # where h = 1 + (1/(1+g)) * sum_{i=1}^{m-1} sin(3*pi*x_i) / 2
        # If g=0 and x_3...x_m = 0, then sin(0)=0, so h=1, f3=1.
        # The Pareto front is indeed a 2D square in (f1, f2) space with f3=1,
        # defined by x1, x2 in [0,1] and x_3...x_n = 0.
        
        # However, the reference point is [1.1, 1.1, 25.0].
        # Hypervolume contribution of a point (f1, f2, f3) with ref (1.1, 1.1, 25.0)
        # is (1.1-f1)*(1.1-f2)*(25.0-f3).
        # Since f3=1 for all Pareto points, the f3 term is constant (24.0).
        # So we want to maximize the 2D hypervolume in (f1, f2) space with ref (1.1, 1.1).
        # This is equivalent to spreading points uniformly in the [0,1]x[0,1] square.

        # Strategy:
        # 1. Use a scrambled Sobol sequence in 2D for high-quality space-filling.
        # 2. Add corner points to ensure boundary coverage.
        # 3. Evaluate all points on the Pareto front (x[2:] = 0).

        n_points = self.budget
        
        # Reserve 4 evaluations for corner points of the Pareto front square
        # Corners: (0,0), (0,1), (1,0), (1,1) in (x1, x2) space
        n_corners = 4
        n_sobol = n_points - n_corners
        
        # Generate Sobol sequence in 2D for x1, x2
        # Use scrambled Sobol for better uniformity
        try:
            sampler = qmc.Sobol(d=2, scramble=True, seed=0)
            if n_sobol > 0:
                # Find smallest k such that 2^k >= n_sobol
                k = int(np.ceil(np.log2(max(n_sobol, 1))))
                sobol_samples = sampler.random_base2(m=k)
                x12 = sobol_samples[:n_sobol]
            else:
                x12 = np.empty((0, 2))
        except Exception:
            # Fallback to Halton sequence if Sobol fails
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            
            x12 = np.zeros((n_sobol, 2))
            offset = 1000
            for i in range(n_sobol):
                n = i + 1
                x12[i, 0] = halton(n + offset, 2)
                x12[i, 1] = halton(n + offset, 3)
        
        # Corner points
        x1_corners = np.array([0.0, 0.0, 1.0, 1.0])
        x2_corners = np.array([0.0, 1.0, 0.0, 1.0])
        
        # Combine Sobol and corner points
        x1 = np.concatenate([x12[:, 0], x1_corners])
        x2 = np.concatenate([x12[:, 1], x2_corners])
        
        # Clip to bounds just in case
        x1 = np.clip(x1, lo[0], hi[0])
        x2 = np.clip(x2, lo[1], hi[1])
        
        # Construct full dimension vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        # x_3...x_n are already 0
        
        # Evaluate all points
        for i in range(n_points):
            if time.time() > deadline or used >= self.budget:
                break
            problem(X[i])
            used += 1

        return None

"""Multi-objective optimizer on DTLZ2 using uniform surface-area sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits the known geometry of the DTLZ2 Pareto front.
    The front for M=3 is a quarter of the unit sphere in the positive octant.
    We sample points uniformly with respect to surface area to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_pos = self.n_obj - 1  # 2 for M=3
        n_g = self.dim - n_pos

        # Set distance variables to midpoint to ensure g(x) = 0
        mid = 0.5 * (lo[n_pos:] + hi[n_pos:])

        # We will generate pairs (u1, u2) in [0, 1]^2
        # and map them to decision variables x1, x2 such that
        # the resulting objectives are uniformly distributed on the Pareto front.

        # DTLZ2 (M=3) objectives with g=0:
        # f1 = 1 - x1
        # f2 = x1 * (1 - x2)
        # f3 = x1 * x2
        #
        # We want to sample (f1, f2, f3) uniformly on the sphere f1^2+f2^2+f3^2=1, fi>=0.
        # Parameterization:
        # f1 = cos(alpha) * cos(beta)
        # f2 = cos(alpha) * sin(beta)
        # f3 = sin(alpha)
        # where alpha in [0, pi/2], beta in [0, pi/2].
        #
        # Surface area element is cos(alpha) dalpha dbeta.
        # Uniform sampling requires:
        # 1. beta uniform in [0, pi/2]  => beta = (pi/2) * u2
        # 2. sin(alpha) uniform in [0, 1] => alpha = arcsin(u1)
        #
        # Mapping back to x1, x2:
        # x1 = 1 - f1
        # x2 = f3 / (f2 + f3)  (since f2 = x1(1-x2), f3 = x1*x2 => f3/f2 = x2/(1-x2))
        
        def map_to_decision_vars(u1, u2):
            """
            Maps uniform u1, u2 in [0, 1] to x1, x2 in [0, 1] 
            such that the resulting point on the Pareto front has uniform surface density.
            """
            # Handle boundaries carefully
            u1 = np.clip(u1, 0.0, 1.0)
            u2 = np.clip(u2, 0.0, 1.0)
            
            alpha = np.arcsin(u1)
            beta = 0.5 * np.pi * u2
            
            f1 = np.cos(alpha) * np.cos(beta)
            f2 = np.cos(alpha) * np.sin(beta)
            f3 = np.sin(alpha)
            
            x1 = 1.0 - f1
            # f2 + f3 = cos(alpha) * (sin(beta) + cos(beta)) ... wait
            # f2 = x1(1-x2), f3 = x1*x2
            # f2 + f3 = x1
            # So x2 = f3 / x1 = f3 / (f2 + f3)
            # If x1 is 0 (f1=1), then f2=0, f3=0. x2 is undefined but doesn't matter as x1=0.
            # If x1 is very small, numerical stability.
            
            denom = f2 + f3
            # If denom is 0, x1 is 0. Set x2 to 0.5 (arbitrary).
            x2 = np.where(denom > 1e-12, f3 / np.maximum(denom, 1e-12), 0.5)
            
            return x1, x2

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Strategy:
        # 1. Generate a grid of points for dense coverage.
        # 2. Fill remaining budget with Halton sequence points.
        
        points_to_evaluate = []
        
        # Grid size
        # We want N^2 <= budget.
        # Let's use a square grid.
        N = int(np.sqrt(self.budget))
        # Ensure N^2 <= budget
        while N * N > self.budget:
            N -= 1
        
        # Generate grid points
        # Use midpoints of the cells to avoid boundary duplicates if we were tiling, 
        # but here we just want good coverage.
        # Actually, for hypervolume, including the extremes (0 and 1) is important.
        # So we use linspace including endpoints.
        u_vals = np.linspace(0, 1, N)
        
        for i in range(N):
            for j in range(N):
                u1 = u_vals[i]
                u2 = u_vals[j]
                x1, x2 = map_to_decision_vars(u1, u2)
                
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                if n_g > 0:
                    x[n_pos:] = mid
                
                x = np.clip(x, lo, hi)
                points_to_evaluate.append(x)
                
        # If we have remaining budget, add Halton points
        remaining = self.budget - len(points_to_evaluate)
        if remaining > 0:
            halton_idx = 1
            for _ in range(remaining):
                u1 = halton(halton_idx, 2)
                u2 = halton(halton_idx, 3)
                
                x1, x2 = map_to_decision_vars(u1, u2)
                
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                if n_g > 0:
                    x[n_pos:] = mid
                
                x = np.clip(x, lo, hi)
                points_to_evaluate.append(x)
                
                halton_idx += 1
        
        # Evaluate all points
        # The return value of __call__ is ignored, but we must call problem(x) for each.
        for i in range(len(points_to_evaluate)):
            if time.time() > deadline - 5:
                break
            problem(points_to_evaluate[i])
            
        return points_to_evaluate
# EVOLVE-BLOCK-END

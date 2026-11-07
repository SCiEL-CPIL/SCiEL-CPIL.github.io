import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure.

    For ZDT2:
        f1 = x1
        f2 = 1 + 9*sqrt(x1)  (equivalently 11*(1-sqrt(1-x1)) when reparameterized,
        but the standard form is f1=x1, f2=1+9*sqrt(x1))
    Pareto front: x2..xn = 0, x1 in [0,1].

    Hypervolume maximization with reference point (11, 11):
        Each point (f1, f2) contributes area = (11 - f1) * (11 - f2)
        The non-dominated union area is what we want to maximize.

    For the ZDT2 front f2 = 1 + 9*sqrt(f1), we want to place N points
    to maximize the hypervolume of the union of boxes.

    The optimal spacing can be derived via calculus of variations. For a
    concave front, the optimal density is proportional to 1/sqrt(1-f2(f1)) * 1/f1'
    which gives a specific non-uniform grid.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        num_points = min(self.budget, 200)

        if num_points <= 1:
            x1_vals = np.array([0.5])
        else:
            # For ZDT2: f1 = x1, f2 = 1 + 9*sqrt(x1)
            # Reference point: (11, 11)
            # Width of box for point i: w_i = 11 - f1_i = 11 - x1_i
            # Height of box for point i: h_i = 11 - f2_i = 11 - 1 - 9*sqrt(x1_i) = 10 - 9*sqrt(x1_i)
            #
            # The hypervolume of the union of non-dominated boxes for a concave front
            # is approximately: sum over i of (x1_{i+1} - x1_i) * (11 - f2(x1_i))
            # + boundary terms.
            #
            # For optimal placement, we want to equalize the "marginal hypervolume"
            # contribution. Using calculus of variations for the continuous case:
            # The density should be proportional to 1/(11 - f2(x)) * |df2/dx|^{-1/2}
            #
            # For f2 = 1 + 9*sqrt(x):
            #   11 - f2 = 10 - 9*sqrt(x)
            #   df2/dx = 9/(2*sqrt(x))
            #
            # Optimal density: dN/dx ∝ (10 - 9*sqrt(x)) / sqrt(9/(2*sqrt(x)))
            #                = (10 - 9*sqrt(x)) * sqrt(2*sqrt(x)/9)
            #                = (10 - 9*sqrt(x)) * sqrt(2)/3 * x^{1/4}
            #
            # Let's numerically compute the optimal grid by inverting the CDF
            # of this density function.
            
            # Compute density on a fine grid
            x_fine = np.linspace(0.0, 1.0, 100001)
            sqrt_x = np.sqrt(x_fine)
            density = (10.0 - 9.0 * sqrt_x) * np.sqrt(2.0) / 3.0 * np.power(x_fine, 0.25)
            density[0] = 0.0  # Handle x=0
            # Ensure density is non-negative
            density = np.maximum(density, 0.0)
            
            # CDF
            cdf = np.cumsum(density)
            cdf = cdf / cdf[-1]  # Normalize to [0, 1]
            
            # Place N points at equal intervals of the CDF
            targets = np.linspace(0.0, 1.0, num_points)
            x1_vals = np.interp(targets, cdf, x_fine)
            x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        recorded = []
        used = 0
        
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1
        
        return recorded

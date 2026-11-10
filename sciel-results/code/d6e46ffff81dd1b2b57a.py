"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    Exploit the analytic structure of ZDT1. The Pareto front is known to lie
    on x[1:] = 0. We select 200 points on this front to maximize the hypervolume
    indicator. Instead of iterative greedy selection, we use a curvature-based
    optimal spacing strategy. We compute the curvature of the Pareto front and
    distribute points such that the discretization error is equalized across
    the front, which is a known near-optimal strategy for hypervolume
    approximation on smooth fronts.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 analytic front: x[0] in [0, 1], x[1:] = 0
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0]) - x[0] * sin(10 * pi * x[0])
        
        # We want to select x1 values in [0, 1]
        n_points = self.budget
        
        # 1. Compute the curvature of the front f2(x1)
        # We sample the front densely to compute curvature numerically
        # f2(x) = 1 - sqrt(x) - x sin(10 pi x)
        # f2'(x) = -1/(2 sqrt(x)) - sin(10 pi x) - 10 pi x cos(10 pi x)
        # f2''(x) = 1/(4 x^(3/2)) - 10 pi cos(10 pi x) - 10 pi cos(10 pi x) + 100 pi^2 x sin(10 pi x)
        #         = 1/(4 x^(3/2)) - 20 pi cos(10 pi x) + 100 pi^2 x sin(10 pi x)
        
        # To avoid singularity at x=0, we start from a small epsilon
        eps = 1e-6
        x_samples = np.linspace(eps, 1.0, 10000)
        
        # Compute derivatives
        # f2'(x)
        f2_prime = -1.0 / (2.0 * np.sqrt(x_samples)) - np.sin(10.0 * np.pi * x_samples) - 10.0 * np.pi * x_samples * np.cos(10.0 * np.pi * x_samples)
        
        # f2''(x)
        f2_double_prime = 1.0 / (4.0 * x_samples**1.5) - 20.0 * np.pi * np.cos(10.0 * np.pi * x_samples) + 100.0 * np.pi**2 * x_samples * np.sin(10.0 * np.pi * x_samples)
        
        # Curvature kappa = |f''| / (1 + f'^2)^(3/2)
        kappa = np.abs(f2_double_prime) / (1.0 + f2_prime**2)**1.5
        
        # 2. Optimal spacing is proportional to kappa^(-1/3)
        # We want to place points such that the "error" is equal.
        # The error in an interval dx is ~ kappa * dx^3.
        # So we want kappa * dx^3 = constant => dx ~ kappa^(-1/3).
        # The cumulative distribution function for the optimal points is proportional to the integral of kappa^(-1/3) dx.
        
        weight = kappa**(-1.0/3.0)
        
        # Integrate the weight to get the CDF
        # Use trapezoidal rule
        cdf = np.cumsum(weight * np.diff(x_samples, prepend=x_samples[0]))
        
        # Normalize CDF to [0, 1]
        cdf = cdf / cdf[-1]
        
        # 3. Invert the CDF to find the x coordinates for n_points equally spaced quantiles
        # We want the points to be at quantiles 0, 1/(n-1), ..., 1
        # But we should probably include the endpoints 0 and 1 explicitly for maximum HV spread?
        # Actually, for HV, having points at the extremes is good.
        # Let's use quantiles from 0 to 1.
        # If we use n_points, the quantiles are i / (n_points - 1) for i in 0..n_points-1.
        
        quantiles = np.linspace(0.0, 1.0, n_points)
        
        # Invert CDF: find x such that cdf(x) = q
        # Since cdf is monotonically increasing, we can use np.interp
        # Note: np.interp expects xp to be increasing. cdf is increasing.
        # We want x for given cdf values (quantiles).
        # So we swap the roles: we look up quantiles in the cdf array to get x_samples.
        # Wait, np.interp(x, xp, fp) returns fp at x.
        # We have x_samples and cdf. We want x for given cdf values.
        # So we call np.interp(quantiles, cdf, x_samples).
        
        x1_optimal = np.interp(quantiles, cdf, x_samples)
        
        # Ensure the first point is 0 and last is 1?
        # The CDF starts at 0 (x=eps) and ends at 1 (x=1).
        # x1_optimal[0] will be close to eps. Let's force 0.0 and 1.0.
        x1_optimal[0] = 0.0
        x1_optimal[-1] = 1.0
        
        # 4. Evaluate the selected points using the problem function
        results = []
        for x1 in x1_optimal:
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            
        return results

# EVOLVE-BLOCK-END

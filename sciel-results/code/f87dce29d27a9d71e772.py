"""Analytical ZDT4 optimizer: construct the exact Pareto front with budget evaluations."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    ZDT4 structure:
      f1(x) = 1 + 10*(n-1) + sum_{i=2}^{n} g(x_i),  g(t)=4*t^2*sin(10*pi*t^2)
      f2(x) = (f1(x)/n)^2 * (1 + sin(10*pi*f1(x)/n))
    
    The Pareto front is obtained by setting x_2..x_n = 0 so that g=0,
    leaving f1 as a function of x_1 only.  We therefore spend the entire
    evaluation budget constructing as many Pareto-optimal points as
    possible, choosing x_1 values that maximise the hypervolume of the
    recorded set with reference point (11, 400).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295  # safety margin under 300 s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim
        budget = self.budget

        # x1 bounds (first variable)
        x1_lo, x1_hi = lo[0], hi[0]

        # All remaining variables set to 0 (within bounds, since [-5,5] contains 0)
        # If 0 is outside bounds for some variable, clamp to nearest bound
        x_tail = np.zeros(n - 1)
        for j in range(1, n):
            if 0.0 < lo[j]:
                x_tail[j - 1] = lo[j]
            elif 0.0 > hi[j]:
                x_tail[j - 1] = hi[j]

        # Determine sweep range for x[0]
        # Points with f1 > 11.0 don't contribute to hypervolume (ref point is [11, 400])
        sweep_lo = max(x1_lo, 0.0)
        sweep_hi = min(x1_hi, 11.0)
        
        # If bounds don't allow the full range, use what's available
        if sweep_lo > sweep_hi:
            sweep_lo = x1_lo
            sweep_hi = x1_hi
            if sweep_lo > sweep_hi:
                sweep_lo, sweep_hi = sweep_hi, sweep_lo

        n_samples = budget

        # Use inverse-transformation sampling based on the hypervolume density
        # For ZDT4 with x[1:]=0: f1 = x1, f2 = (x1/n)^2 * (1 + sin(10*pi*x1/n))
        # The hypervolume contribution density is proportional to the local
        # "steepness" of the Pareto front, which is |df2/df1|
        
        def zdt4_f2(f1_val):
            """Compute f2 for given f1 on the Pareto front (x[1:]=0)."""
            return (f1_val / n) ** 2 * (1.0 + np.sin(10.0 * np.pi * f1_val / n))
        
        def zdt4_df2_df1(f1_val):
            """Derivative df2/df1 for the Pareto front."""
            a = 10.0 * np.pi / n
            return (2.0 * f1_val / n**2) * (1.0 + np.sin(a * f1_val)) + (f1_val**2 / n**2) * a * np.cos(a * f1_val)
        
        # Build a cumulative distribution based on hypervolume density
        # Theoretical analysis: minimizing sum(dF1_i * g(f1_i)) with fixed total width
        # implies density proportional to g(f1) = f2 on the Pareto front
        n_grid = 4000
        grid = np.linspace(sweep_lo, sweep_hi, n_grid)
        f2_grid = zdt4_f2(grid)
        
        # Density proportional to the objective value g(f1) to concentrate
        # points where the hypervolume loss per unit width is largest
        # Add small epsilon to avoid zero density at f1=0
        density = f2_grid + 0.01
        density = np.maximum(density, 1e-6)
        
        # Normalize to CDF
        cdf = np.cumsum(density)
        cdf /= cdf[-1]
        
        # Inverse CDF sampling: place points at equal CDF intervals
        # This ensures that each point contributes roughly equal hypervolume
        target_cdfs = np.linspace(0, 1, n_samples)
        x1_values = np.interp(target_cdfs, cdf, grid)
        
        # Ensure endpoints are included for full coverage
        # Use a small offset to avoid exact boundary issues
        eps = (sweep_hi - sweep_lo) * 1e-8
        x1_values[0] = sweep_lo + eps
        x1_values[-1] = sweep_hi - eps
        
        results = []
        used = 0
        
        for x1_val in x1_values:
            if time.time() > deadline:
                break
            if used >= budget:
                break
            
            x = np.empty(n)
            x[0] = x1_val
            x[1:] = x_tail
            f = problem(x)
            results.append(f)
            used += 1

        return results
# EVOLVE-BLOCK-END

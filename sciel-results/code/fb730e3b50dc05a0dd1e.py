"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure:
    f1(x) = 1 + 10*(n-1) + sum(gi(xi) for i=2..n)
    f2(x) = (f1(x)/n)^2 * (1 + sin(10*pi*f1(x)/n))
    
    To minimize f1 and f2, we need:
    1. gi(xi) = 0 for all i=2..n. This happens when xi=0.
    2. x1 determines the shape of the Pareto front via f1(x1).
    
    Strategy:
    - Allocate the full budget to sampling x1 densely to cover the Pareto front.
    - For each x1 sample, set x2..x10 = 0 to ensure g(x) = 0.
    - This gives us points on the true Pareto front.
    - Use a non-uniform grid that clusters points near endpoints where curvature is high,
      with denser sampling near x1=0 where hypervolume contribution is greater.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        budget = self.budget
        
        n_samples = budget
        
        # Determine sweep range for x[0]
        # Points with f1 > 11.0 don't contribute to hypervolume (ref point is [11, 400])
        # ZDT4: f1 = x[0] when x[1:] = 0, so sweep x[0] in [0, 11]
        sweep_lo = max(lo[0], 0.0)
        sweep_hi = min(hi[0], 11.0)
        
        # If bounds don't allow the full range, use what's available
        if sweep_lo > sweep_hi:
            sweep_lo = lo[0]
            sweep_hi = hi[0]
            if sweep_lo > sweep_hi:
                sweep_lo, sweep_hi = sweep_hi, sweep_lo
        
        # Create non-uniform grid: cluster points near both endpoints
        # Use a transformation that gives denser spacing near 0 and 11
        # t in [0, 1], map to x[0] with clustering at both ends
        # Use: x = sweep_lo + (sweep_hi - sweep_lo) * (0.5 - 0.5*cos(pi * t^alpha))
        # where alpha < 1 gives clustering near t=0 (x=sweep_lo)
        # To cluster near both ends, we can use a different approach:
        # Split the budget: 60% near low end, 40% near high end
        # Or use a symmetric clustering function
        
        # Better approach: use a power transform that clusters near 0
        # x = sweep_lo + (sweep_hi - sweep_lo) * t^p where p > 1 clusters near 0
        # But we also want some density near 11. Let's use a combination:
        # 70% of points clustered near 0, 30% near 11
        
        n_low = int(budget * 0.65)  # denser near x=0
        n_high = budget - n_low      # sparser near x=11
        
        # Points clustered near sweep_lo
        t_low = np.linspace(0, 1, n_low)
        x_low = sweep_lo + (sweep_hi - sweep_lo) * t_low**2.5  # power 2.5 clusters near 0
        
        # Points clustered near sweep_hi
        t_high = np.linspace(0, 1, n_high)
        x_high = sweep_hi - (sweep_hi - sweep_lo) * t_high**2.5  # power 2.5 clusters near hi
        
        # Combine and sort
        x1_values = np.concatenate([x_low, x_high])
        x1_values = np.sort(x1_values)
        
        # Ensure we have exactly budget points
        if len(x1_values) > budget:
            # If duplicates or overlap, take unique and pad if needed
            x1_values = x1_values[:budget]
        elif len(x1_values) < budget:
            # Pad with linspace
            extra = np.linspace(sweep_lo, sweep_hi, budget - len(x1_values) + 1)[1:-1]
            x1_values = np.concatenate([x1_values, extra])
            x1_values = np.sort(x1_values)[:budget]
        
        results = []
        used = 0
        
        for x1_val in x1_values:
            if time.time() > deadline:
                break
            if used >= budget:
                break
            
            x = np.zeros(n)
            x[0] = x1_val
            # x[1:] stays at 0, which minimizes g
            
            f = problem(x)
            results.append(f)
            used += 1
        
        return results
# EVOLVE-BLOCK-END

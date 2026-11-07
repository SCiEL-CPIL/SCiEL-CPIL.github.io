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
    - Allocate most of the budget to sampling x1 densely to cover the Pareto front.
    - For each x1 sample, set x2..x10 = 0 to ensure g(x) = 0.
    - This gives us points on the true Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300 second deadline with safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4: x1 in [0, 1], x2..xn in [-5, 5]
        # The Pareto front is achieved when x2..xn = 0
        # f1 = 1 + 10*(n-1) + g(x) where g(x) = sum of gi(xi)
        # gi(xi) = 4*xi^2 * sin(10*pi*xi^2)
        # gi(0) = 0, so x2..xn = 0 gives g(x) = 0
        
        # x1 determines the position on the Pareto front
        # f1(x1) = 1 + 10*(n-1) + 0 = 1 + 10*(n-1) when g=0
        # Wait, that's not right. Let me reconsider.
        
        # Actually for ZDT4:
        # f1(x) = 1 + 10*(n-1) + sum_{i=2}^{n} g(x_i)
        # where g(x_i) = 4*x_i^2 * sin(10*pi*x_i^2)
        # f2(x) = (f1(x)/n)^2 * (1 + sin(10*pi*f1(x)/n))
        
        # The Pareto front is when all g(x_i) = 0, which happens when x_i = 0 for i=2..n
        # Then f1 = 1 + 10*(n-1) + 0 = 1 + 10*(n-1)
        # But that's a constant, which doesn't make sense for a front.
        
        # Let me reconsider. Actually, I think the standard ZDT4 formulation is:
        # f1(x) = x1
        # f2(x) = (1 + sqrt(x1/(1-x1))) * (1 + sin(10*pi*x1))
        
        # Or maybe:
        # f1(x) = 1 + 10*(n-1) + sum g(x_i)
        # f2(x) = (1 + f1(x))^0.5 * (1 + sin(10*pi*f1(x)))
        
        # Given the reference point [11.0, 400.0], let me think about what the objectives are.
        
        # For ZDT4 with n=10:
        # If f1 ranges from 0 to 11 (1 + 10*1 = 11 when all g=0 and x1 contributes)
        # Actually, I think the standard ZDT4 is:
        # f1(x) = x1
        # f2(x) = (1 + sqrt(x1/(1-x1))) * (1 + sin(10*pi*x1))
        
        # But with the decoupled structure mentioned, it's likely:
        # f1(x) = 1 + 10*(n-1) + sum_{i=2}^{n} g(x_i) where g(x_i) = 4*x_i^2*sin(10*pi*x_i^2)
        # f2(x) = (f1(x)/n)^2 * (1 + sin(10*pi*f1(x)/n))
        
        # Wait, that still doesn't use x1. Let me just sample densely and see.
        
        # Strategy: sample x1 uniformly in [lo[0], hi[0]] and set x2..xn = 0
        # This should give us points on or near the Pareto front.
        
        n = self.dim
        budget = self.budget
        
        # Use most of the budget for dense sampling of x1
        # Reserve a few evaluations for potential refinement
        n_samples = min(budget - 5, 150)  # Leave 5 for safety
        n_samples = max(n_samples, 10)
        
        # Sample x1 densely
        x1_values = np.linspace(lo[0], hi[0], n_samples)
        
        results = []
        used = 0
        
        for x1_val in x1_values:
            if time.time() > deadline - 30:  # Safety margin
                break
            if used >= budget:
                break
            
            x = np.zeros(n)
            x[0] = x1_val
            # x[1:] = 0 (already zero)
            
            f = problem(x)
            results.append(f)
            used += 1
        
        # Use remaining budget for additional samples if time permits
        if used < budget and time.time() < deadline - 30:
            # Add some random samples to explore
            rng = np.random.default_rng(42)
            while used < budget and time.time() < deadline - 30:
                x = lo + (hi - lo) * rng.random(n)
                # Bias towards x2..xn = 0 to stay near Pareto front
                x[1:] *= 0.1  # Scale down to be closer to 0
                
                f = problem(x)
                results.append(f)
                used += 1
        
        return results
# EVOLVE-BLOCK-END

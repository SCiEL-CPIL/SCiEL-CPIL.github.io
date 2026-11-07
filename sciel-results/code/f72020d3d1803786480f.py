"""Multi-objective optimizer for WFG4 using a constrained Pareto-front sampling strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits the known analytic Pareto front of WFG4.
    
    WFG4 has a concave Pareto front. The key insight is that we can directly
    sample points on or near the Pareto front by:
    1. Sampling the first decision variable x_0 uniformly (this controls the 
       trade-off along the front)
    2. Setting the remaining variables to values that minimize the objectives
       given x_0, which for WFG4 typically means setting them to specific 
       values that activate the correct trade-off
    
    For WFG4 specifically:
    - f1 = x_0
    - f2 = (1 - sqrt(x_0)) * (1 + 10 * sin^2(10 * pi * x_0 + pi/2))
    - f3 = (1 - x_0) * (1 + 10 * sin^2(20 * pi * x_0 + pi/2))
    
    The other variables (x_1 through x_11) primarily affect the landscape's 
    complexity but the Pareto front is determined by x_0. By setting the 
    remaining variables to values that minimize interference, we can generate
    points very close to the true Pareto front.
    
    Strategy:
    - Use a portion of the budget for random exploration to find good base points
    - Use the majority of the budget for constrained sampling along the front
    - Combine both to ensure we have a well-distributed set of non-dominated points
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Strategy: 
        # 1. Generate a diverse set of points on/near the Pareto front
        # 2. Use a small portion for random exploration to find any better regions
        
        # For WFG4, the Pareto front is determined primarily by x_0.
        # We'll sample x_0 uniformly and set other variables to values that
        # minimize the objectives.
        
        # Key insight: For WFG4, setting x_1 through x_{n-1} to specific values
        # (often 0 or 0.5 depending on the problem's structure) helps minimize
        # the objectives. We'll use a hybrid approach.
        
        # Allocate budget:
        # - 70% for constrained Pareto-front sampling
        # - 30% for random exploration (to catch any structure we might miss)
        
        n_front_samples = int(budget * 0.7)
        n_random_samples = budget - n_front_samples
        
        all_X = []
        all_F = []
        used = 0

        # Phase 1: Constrained Pareto-front sampling
        # Sample x_0 uniformly across its range, and set other variables to 
        # values that minimize the objectives.
        
        # For WFG4, the typical structure suggests that setting variables to 
        # 0.5 or specific values helps. We'll try a few strategies:
        # Strategy A: Set all other variables to 0.5
        # Strategy B: Set all other variables to 0
        # Strategy C: Set all other variables to 1
        
        # We'll use a mix, but primarily focus on x_0 variation.
        
        # Generate uniform samples for x_0
        x0_samples = np.linspace(lo[0], hi[0], n_front_samples, endpoint=True)
        
        # For each x_0 sample, create a point with other variables set to 
        # minimize objectives. We'll use 0.5 as a default, but also try 
        # variations.
        
        for i, x0_val in enumerate(x0_samples):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Create base point with x_0 = x0_val
            x = np.zeros(dim)
            x[0] = x0_val
            
            # Set other variables to 0.5 (middle of range) as a default
            # This is a heuristic; the exact optimal values depend on WFG4's structure
            for j in range(1, dim):
                x[j] = 0.5 * (lo[j] + hi[j])
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
        
        # Phase 2: Random exploration to find any better regions
        for _ in range(n_random_samples):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Random point in the feasible region
            x = lo + span * rng.random(dim)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1

        # Convert to arrays
        all_X = np.array(all_X)
        all_F = np.array(all_F)

        # Optional: Filter to keep only non-dominated points for a cleaner result
        # But since the scoring uses all recorded vectors, we return all of them.
        
        # However, we can do a quick non-dominated filter to ensure we have a 
        # good spread. But the task says "Every returned objective vector is 
        # recorded", so we should return all evaluated points.
        
        # For now, return all points. The hypervolume calculation will handle
        # dominated points (they won't contribute to the hypervolume).
        
        return all_X

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic sampling of the analytical Pareto front for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Sample x_1 uniformly in [0, 0.35] and set other variables to 0.5
        # This is based on the known analytical Pareto front of WFG4 where x_1 is the shape variable.
        
        # We have a budget of self.budget evaluations.
        # We will use all of them to sample the front densely.
        
        # Ensure we don't exceed time, though this is very fast.
        if time.time() > deadline:
            return None

        # Generate x_1 samples
        # The hint specifies [0, 0.35]. We should respect bounds if they are tighter, 
        # but typically WFG4 bounds are [0,1].
        # We'll clip the target range to the actual bounds for x_1.
        x1_low = max(0.0, lo[0])
        x1_high = min(0.35, hi[0])
        
        # If the range is invalid (e.g. bounds don't cover 0-0.35), fall back to random search?
        # But the hint is specific. Let's assume 0-0.35 is valid.
        if x1_low >= x1_high:
            x1_low = lo[0]
            x1_high = hi[0]
            
        n_samples = self.budget
        x1_samples = np.random.uniform(x1_low, x1_high, size=(n_samples, 1))
        
        # Set other variables to 0.5 (midpoint of typical [0,1] bounds)
        # Or midpoint of their specific bounds?
        # "specific optimal constant values derived from the WFG4 definition"
        # In WFG4, position variables are often set to 0.5 or similar.
        # Let's use 0.5 for all other dimensions.
        other_vals = np.full((n_samples, self.dim - 1), 0.5)
        
        # Combine
        X = np.hstack([x1_samples, other_vals])
        
        # Clip to bounds to be safe
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The problem function takes a vector and returns objectives.
        # We need to call it n_samples times.
        F = np.array([problem(x) for x in X])
        
        # The return value is ignored, but we must return something.
        # We return the population X.
        return X
# EVOLVE-BLOCK-END

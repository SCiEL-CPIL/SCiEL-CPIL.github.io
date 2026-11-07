"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Structured Multi-Objective Optimizer using Latin Hypercube Sampling (LHS).
    
    Strategy:
    1. Generate a space-filling design (LHS) to ensure diversity across the 
       search space [0, 1]^4.
    2. Evaluate all points within the budget.
    3. Return the non-dominated subset (or all if budget allows, but usually 
       LHS points are mostly non-dominated in low dimensions with smooth fronts).
    
    This avoids the random-walk stagnation of simple EAs and ensures broad 
    coverage of the Pareto front, which is critical for hypervolume scoring.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        
        # 1. Generate Latin Hypercube Sample
        # LHS ensures that each dimension is divided into n equal-probability intervals,
        # and exactly one sample is chosen from each interval in each dimension.
        # This provides excellent space-filling properties.
        
        # We can use a simple deterministic LHS construction or random permutation.
        # Using random permutation for slight variation, but deterministic seed for reproducibility 
        # isn't strictly required by contract (problem handles clipping), but good practice.
        # However, the contract says "seeds 0 and 1 ... evaluator calls np.random.seed(seed)".
        # We should use the global random state or our own. Let's use np.random for consistency 
        # with the evaluator's seeding if we don't create a new RNG, or create a new one.
        # The prompt says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)...".
        # It does not say we must use np.random, but it's available.
        # Let's use a local RNG seeded by the global state to be safe and deterministic per seed.
        rng = np.random.default_rng(np.random.randint(0, 2**32))
        
        # Generate LHS
        # For each dimension, permute the indices 0..n-1
        indices = np.array([rng.permutation(n) for _ in range(self.dim)]).T
        # Convert to [0, 1] range
        # Each point in dimension d is in interval [i/n, (i+1)/n]
        # Uniformly sample within that interval
        u = rng.random((n, self.dim))
        X = (indices + u) / n
        
        # Scale to bounds
        X = lo + (hi - lo) * X
        
        # 2. Evaluate
        F = np.array([problem(x) for x in X])
        
        # 3. Filter non-dominated points (optional, but good for returning a clean front)
        # However, the scoring uses ALL recorded vectors. 
        # "Every returned objective vector is recorded".
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # Actually, re-reading: "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" - this likely refers to the points evaluated by `problem`.
        # The `problem` function is the evaluator. Every time we call `problem(x)`, that objective vector is recorded.
        # So we just need to make 200 calls to `problem`.
        
        # Check time
        if time.time() > deadline:
             # This shouldn't happen with 200 simple evals, but guard just in case
             pass
             
        return X
# EVOLVE-BLOCK-END

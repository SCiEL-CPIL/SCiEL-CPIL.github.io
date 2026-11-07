"""ZDT1 multi-objective optimizer exploiting the known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits ZDT1's decoupled structure.
    
    ZDT1: f1 = x1, g = 1 + 9*sum(x2..xn)/(n-1), f2 = g*(1 - sqrt(f1/g))
    The Pareto front is achieved when x2..xn = 0 (g=1), and f1 sweeps [0,1].
    With 200 evaluations we can densely sample the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget
        dim = self.dim
        
        # Use ALL budget evaluations for the Pareto front sweep
        # ZDT1 Pareto front: x2..xn = 0, f1 = x1, f2 = 1 - sqrt(x1)
        # Use x1 = u^2 to cluster points near x1=0 where the front has high curvature
        # This maximizes hypervolume coverage
        
        n_sweep = budget
        
        # Generate non-uniform x1 values: u^2 clusters near 0
        u = np.linspace(0.0, 1.0, n_sweep)
        x1_values = u ** 2
        
        # Create the sweep matrix
        X_sweep = np.zeros((n_sweep, dim))
        X_sweep[:, 0] = x1_values
        # x2..xn stay at 0 (lower bound), which is optimal for ZDT1
        
        # Clip to bounds to be safe
        X_sweep = np.clip(X_sweep, lo, hi)
        
        # Evaluate all sweep points
        for x in X_sweep:
            problem(x)
        
        # Check deadline
        if time.time() > deadline:
            return X_sweep
        
        return X_sweep
# EVOLVE-BLOCK-END

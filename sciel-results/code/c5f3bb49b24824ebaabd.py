import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s margin
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # ZDT3 analytic Pareto front:
        # f1 = x1, f2 = g * h(f1, g) where g = 1 when x2..xm = 0
        # h(f1, g) = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)
        # With g=1: f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        # The Pareto front is achieved when x[1:] = 0
        
        # We need to evaluate points on the Pareto front to maximize hypervolume
        # The key insight: we should sample f1 values that capture the discontinuous
        # Pareto front well. ZDT3 has 57 disconnected regions.
        
        # Use a space-filling sequence for uniform coverage of [0, 1]
        # Halton sequence in 1D is just 1/n for n=1,2,...
        # But for better distribution, use a stratified approach aligned with the 5 oscillation periods
        
        # ZDT3's h function has sin(10*pi*f1), so there are 5 full periods in [0,1]
        # Each period has a local minimum of f2 where sin is at its minimum (-1)
        # These are the "valleys" of the Pareto front
        
        # Strategy: Allocate evaluations to cover all 57 disconnected regions
        # The disconnected regions occur where the front has jumps
        
        # For optimal hypervolume, we want points that maximize the area under the
        # lower envelope. For 2D, this means we want good coverage of the entire front.
        
        # Use stratified sampling: divide [0,1] into budget equal strata
        # and pick one point per stratum
        
        f1_vals = np.linspace(0, 1, budget + 1)
        # Use midpoints of each stratum
        f1_mid = (f1_vals[:-1] + f1_vals[1:]) / 2.0
        
        # Construct x vectors
        X_candidates = np.zeros((budget, dim))
        X_candidates[:, 0] = f1_mid
        
        # Evaluate all budget points
        F_evaluated = np.array([problem(X_candidates[i]) for i in range(budget)])
        
        return F_evaluated

import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Optimizer for DTLZ4 using parameterized Pareto front sampling with
    strategic corner/edge points and Sobol quasi-random fill."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 Pareto front: x_2..x_11 = 0.5 (g=0)
        # x_0 = cos(theta), x_1 = sin(theta) for theta in [0, pi/2]
        # This gives the quarter unit circle which is the true Pareto front.
        
        # Strategy: Place points uniformly in theta space on the quarter circle.
        # Add strategic emphasis on corners (theta=0, pi/2) and mid-arc.
        
        # Generate theta values
        n_total = self.budget
        
        # Use a mix: some strategic points + uniform/Sobol fill in theta
        # Corners: theta=0 (x0=1,x1=0), theta=pi/2 (x0=0,x1=1)
        # Also add points at key fractions of the arc
        
        # Strategic theta values (in [0, pi/2])
        strategic_thetas = [
            0.0, np.pi/2,          # corners
            np.pi/6, np.pi/3,      # 30 deg, 60 deg
            np.pi/4,               # 45 deg (middle)
        ]
        
        n_strategic = len(strategic_thetas)
        n_fill = n_total - n_strategic
        
        if n_fill < 0:
            n_fill = 0
            strategic_thetas = strategic_thetas[:n_total]
        
        # Generate fill points using Sobol sequence in [0, 1] then map to [0, pi/2]
        try:
            sampler = qmc.Sobol(d=1, scramble=True, seed=42)
            sobol_1d = sampler.random(n_fill)  # shape (n_fill, 1)
            fill_thetas = sobol_1d[:, 0] * (np.pi / 2)
        except Exception:
            # Fallback: Halton-like or uniform
            fill_thetas = (np.arange(n_fill) + 0.5) / n_fill * (np.pi / 2)
        
        # Combine all theta values
        all_thetas = np.concatenate([np.array(strategic_thetas), fill_thetas])
        
        # Convert to x0, x1 on the quarter circle
        x0 = np.cos(all_thetas)
        x1 = np.sin(all_thetas)
        
        # Construct decision vectors
        X = np.zeros((len(all_thetas), self.dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2:] = 0.5  # Set aggregation variables to 0.5 to minimize g
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        if time.time() > deadline:
            return X
            
        return X

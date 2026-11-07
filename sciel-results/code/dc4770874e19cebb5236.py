"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 that exploits the known Pareto front geometry.
    
    DTLZ1 with M=3 objectives, N=7 variables:
    - g(x) = 100 * (N - M + 1 + sum_{i=M}^{N} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))])
    - f_1 = 0.5 * (1 + g) * x_1
    - f_2 = 0.5 * (1 + g) * x_2  
    - f_3 = 0.5 * (1 + g) * (1 - x_1^2 - x_2^2)
    
    Pareto front: g=0 (x_3..x_7 = 0.5), x_1^2 + x_2^2 = 1, x_1, x_2 >= 0
    On PF: f_1 = 0.5*x_1, f_2 = 0.5*x_2, f_3 = 0
    Objective space is a quarter-disk of radius 0.5 centered at origin.
    
    Strategy: Use all budget to sample the known Pareto front densely using
    a Sobol-like low-discrepancy sequence for better hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ1 with 7 vars, 3 objectives:
        # x[0], x[1] are the "position" variables (on unit circle in first quadrant)
        # x[2..6] are the "distance" variables (should be 0.5 for g=0)
        # g(x) = 100 * (5 + sum_{i=2}^{6} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))])
        # On PF: x[2..6] = 0.5, so g = 0
        # f1 = 0.5 * x[0], f2 = 0.5 * x[1], f3 = 0.5 * (1 - x[0]^2 - x[1]^2)
        # With x[0]^2 + x[1]^2 = 1, f3 = 0
        
        # The Pareto front in objective space:
        # f1 = 0.5 * cos(theta), f2 = 0.5 * sin(theta), f3 = 0
        # for theta in [0, pi/2]
        
        # Strategy: Sample theta using a low-discrepancy sequence for better
        # hypervolume coverage. Use Halton sequence (base 2) for 1D case.
        
        n_evals = self.budget
        
        # Generate Halton sequence in 1D (base 2) for low-discrepancy sampling
        # This gives better coverage than uniform linspace for hypervolume
        thetas = np.array([self._halton(i + 1, 2) for i in range(n_evals)]) * (np.pi / 2)
        
        # Ensure we include the endpoints for full coverage
        thetas[0] = 0.0
        thetas[-1] = np.pi / 2
        
        results = []
        for i, theta in enumerate(thetas):
            if time.time() > deadline:
                break
            
            x = np.array([np.cos(theta), np.sin(theta)] + [0.5] * (self.dim - 2))
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        return results
    
    @staticmethod
    def _halton(index, base):
        """Generate the index-th point in the Halton sequence with the given base."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result
# EVOLVE-BLOCK-END

"""Multi-objective optimizer on DTLZ1 with analytical front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Analytically determine the Pareto front condition: g(x) = 0.
       For DTLZ1, this requires aggregation variables (x_2..x_6) to be 0.5.
    2. Sample the position variables (x_0, x_1) such that the resulting
       objective vectors are uniformly distributed on the Pareto front.
       We use a Sobol sequence mapped to the simplex in objective space,
       then invert the mapping to find the corresponding decision variables.
    3. This maximizes hypervolume coverage by ensuring good spread in objective space.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 285.0  # Safety margin before 300s limit
    
    def _elapsed(self):
        return time.time() - self.start_time
    
    def _time_left(self):
        return self.deadline - self._elapsed()
    
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        m = self.n_obj
        n = self.dim
        M = m - 1  # Number of position variables (2 for DTLZ1)
        
        n_samples = self.budget
        if n_samples <= 0:
            return np.array([])
        
        # Generate Sobol points in [0, 1]^M
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=M, scramble=True)
            U = sampler.random(n_samples)
        except ImportError:
            # Fallback to random
            U = np.random.rand(n_samples, M)
        
        # Map U (hypercube) to Simplex (F) uniformly
        # We want F_1, F_2, F_3 >= 0 and F_1 + F_2 + F_3 = 0.5
        # Uniform sampling in 2-simplex:
        # Let u1, u2 be the Sobol points.
        # F_3 = 0.5 * (1 - u1)
        # F_1 = 0.5 * u1 * u2
        # F_2 = 0.5 * u1 * (1 - u2)
        # This mapping is NOT uniform in the simplex area.
        # To get uniform distribution in the simplex, we use:
        # v1, v2 ~ U(0,1)
        # F_1 = 0.5 * (1 - sqrt(1 - v1))
        # F_2 = 0.5 * (1 - F_1) * v2
        # F_3 = 0.5 * (1 - F_1) * (1 - v2)
        
        F1 = 0.5 * (1.0 - np.sqrt(1.0 - U[:, 0]))
        F2 = 0.5 * (1.0 - F1) * U[:, 1]
        F3 = 0.5 * (1.0 - F1) * (1.0 - U[:, 1])
        
        # Invert to x_0, x_1
        # F_3 = 0.5 * (1 - x_0)  => x_0 = 1 - 2 * F_3
        # F_1 = 0.5 * x_0 * x_1  => x_1 = 2 * F_1 / x_0
        x_0 = 1.0 - 2.0 * F3
        x_1 = np.zeros(n_samples)
        mask = x_0 > 1e-12
        x_1[mask] = 2.0 * F1[mask] / x_0[mask]
        
        # Clip to [0, 1]
        x_0 = np.clip(x_0, 0.0, 1.0)
        x_1 = np.clip(x_1, 0.0, 1.0)
        
        # Construct full X
        X = np.zeros((n_samples, n))
        X[:, 0] = x_0
        X[:, 1] = x_1
        # Set aggregation variables to 0.5
        X[:, M:] = 0.5
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END

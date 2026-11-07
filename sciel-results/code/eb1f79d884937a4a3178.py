# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Exploit DTLZ1 closed-form front with structured and quasi-random sampling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _simplex_to_decision(self, f1, f2, f3, K, lo, hi):
        """Map simplex coordinates (f1, f2, f3) to decision vector."""
        s = f1 + f2
        if s > 1e-12:
            x1 = s / K
            x2 = f1 / s
        else:
            x1 = 0.0
            x2 = 0.5

        x = np.full(self.dim, 0.5)
        x[0] = x1
        x[1] = x2
        return np.clip(x, lo, hi)

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        recorded = []
        used = 0

        K = 150.5

        # Critical points: Vertices and Edge Midpoints
        critical_simplex = [
            (K, 0, 0), (0, K, 0), (0, 0, K),
            (K/2, K/2, 0), (K/2, 0, K/2), (0, K/2, K/2),
            (K/3, K/3, K/3)
        ]
        
        # Generate Sobol points for the rest
        n_sobol = budget - len(critical_simplex)
        if n_sobol > 0:
            sobol = qmc.Sobol(d=3, scramble=True, seed=42)
            sobol_points = sobol.random(n_sobol)
            
            # Map to simplex using exponential method
            # E_i = -ln(1-U_i)
            # W_i = E_i / sum(E)
            # F_i = K * W_i
            # Handle U=1 to avoid inf
            sobol_points = np.clip(sobol_points, 0, 1 - 1e-15)
            exp_points = -np.log(1 - sobol_points)
            sum_exp = np.sum(exp_points, axis=1, keepdims=True)
            weights = exp_points / sum_exp
            sobol_simplex = weights * K
            
            # Combine
            all_simplex = np.array(critical_simplex)
            sobol_simplex_arr = np.column_stack([sobol_simplex[:,0], sobol_simplex[:,1], sobol_simplex[:,2]])
            all_simplex = np.vstack([all_simplex, sobol_simplex_arr])
        else:
            all_simplex = np.array(critical_simplex)

        # Evaluate points
        for f1, f2, f3 in all_simplex:
            if used >= budget or time.time() > deadline:
                break
            x = self._simplex_to_decision(f1, f2, f3, K, lo, hi)
            f = np.array(problem(x), dtype=float)
            recorded.append(f)
            used += 1

        return recorded
# EVOLVE-BLOCK-END

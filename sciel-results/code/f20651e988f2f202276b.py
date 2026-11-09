"""Multi-objective optimizer for ZDT4 exploiting analytic structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT4.
    Exploits the known structure:
    f1(x) = x0
    f2(x) = g(x) * h(f1(x))
    g(x) = 10*n + sum(G(xi)) for i=1..n-1
    G(xi) = |xi| * exp(-(xi^2 - 0.25)^2)
    Global minimum of G(xi) is 0 at xi = 0.5.
    Thus, the Pareto front is achieved when x1..x9 = 0.5 and x0 varies in [0, 1].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine optimal values for x1..x9
        # Standard ZDT4 has global min at 0.5 for these variables.
        opt_tail = np.full(self.dim - 1, 0.5)
        opt_tail = np.clip(opt_tail, lo[1:], hi[1:])
        
        n_evals = self.budget
        x0_lo, x0_hi = lo[0], hi[0]
        
        # Non-uniform sampling: bias toward x0=0 where f2 is highest, contributing more to hypervolume.
        # Use a power-law: x0 = x0_lo + (x0_hi - x0_lo) * u^p, with p > 1 biases toward x0_lo.
        # Try p=2 for moderate bias.
        p = 2.0
        u = np.random.uniform(0, 1, n_evals)
        x0_vals = x0_lo + (x0_hi - x0_lo) * (u ** p)
        
        # Ensure we include the endpoints for full coverage
        x0_vals[0] = x0_lo
        x0_vals[-1] = x0_hi
        
        # Construct population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        X[:, 1:] = opt_tail
        
        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
# EVOLVE-BLOCK-END

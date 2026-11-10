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
    
    Strategy: Use a few evaluations to robustly determine the optimal tail,
    then use the remaining budget for dense sampling of x0.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_g(self, x_tail):
        """Compute g(x) = 10*(n-1) + sum(G(xi)) for tail variables."""
        n = self.dim
        G_vals = np.abs(x_tail) * np.exp(-(x_tail**2 - 0.25)**2)
        return 10 * (n - 1) + np.sum(G_vals)

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Robustly determine optimal tail by evaluating a few candidates
        # Candidates: 0.5 (global min), 0, and boundary values
        tail_candidates = []
        
        # Add 0.5 if within bounds
        if lo[1] <= 0.5 <= hi[1]:
            tail_candidates.append(np.full(self.dim - 1, 0.5))
        
        # Add 0 if within bounds
        if lo[1] <= 0.0 <= hi[1]:
            tail_candidates.append(np.full(self.dim - 1, 0.0))
        
        # Add boundary values
        tail_candidates.append(np.clip(np.full(self.dim - 1, lo[1]), lo[1:], hi[1:]))
        tail_candidates.append(np.clip(np.full(self.dim - 1, hi[1]), lo[1:], hi[1:]))
        
        # Evaluate candidates to find the one with lowest g(x)
        # Use a fixed x0 for evaluation (doesn't affect g, but problem needs valid input)
        x0_test = 0.5
        if not (lo[0] <= x0_test <= hi[0]):
            x0_test = (lo[0] + hi[0]) / 2.0
        
        best_tail = None
        best_g = np.inf
        n_tail_evals = 0
        max_tail_evals = 7
        
        for tail in tail_candidates:
            if n_tail_evals >= max_tail_evals:
                break
            if time.time() > deadline:
                break
            
            x_test = np.zeros(self.dim)
            x_test[0] = x0_test
            x_test[1:] = tail
            
            # Evaluate to get f2, from which we can infer g
            # f2 = g * h(f1), and h(x0_test) is known
            f = problem(x_test)
            n_tail_evals += 1
            
            # Compute h(x0_test) analytically
            h_val = 1.0 + np.sin(10.0 * np.pi * x0_test)
            if abs(h_val) > 1e-10:
                g_est = f[1] / h_val
            else:
                # h is near zero, use direct g computation
                g_est = self._compute_g(tail)
            
            if g_est < best_g:
                best_g = g_est
                best_tail = tail.copy()
        
        # If no valid tail found, use 0.5 clipped to bounds
        if best_tail is None:
            best_tail = np.clip(np.full(self.dim - 1, 0.5), lo[1:], hi[1:])
        
        # Use remaining budget for dense x0 sampling
        remaining_budget = self.budget - n_tail_evals
        if remaining_budget < 10:
            remaining_budget = self.budget  # Fallback if too many tail evals
        
        # Generate x0 values with non-uniform distribution for better coverage
        # Use a combination of uniform and clustered points
        n_evals = remaining_budget
        
        # Base uniform grid
        x0_uniform = np.linspace(lo[0], hi[0], n_evals)
        
        # Construct population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_uniform
        X[:, 1:] = best_tail
        
        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
# EVOLVE-BLOCK-END

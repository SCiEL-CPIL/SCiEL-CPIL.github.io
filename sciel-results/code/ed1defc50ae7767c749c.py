"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ1.
    Strategy:
    1. The objective function g(x) for DTLZ1 is minimized at x_i = 0.5 (center of bounds).
    2. The Pareto front is determined by the diversity variable x_d.
    3. We initialize with points at the center of the convergence variables to minimize g.
    4. We spread x_d to cover the diversity.
    5. We use a local search to further minimize g if possible.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0  # Safety margin from 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        
        # DTLZ1: g(x) = sum_{i=1}^{d-1} (x_i - 0.5)^2 (assuming bounds [0,1])
        # Min g is at x_i = 0.5.
        # We allocate most of the budget to finding the best g.
        
        # 1. Initialize with points that have optimal convergence variables
        #    and diverse diversity variables.
        #    Number of initial points: 10
        n_init = 10
        X = np.zeros((n_init, self.dim))
        # Set convergence variables to center
        X[:, :self.dim-1] = center[:self.dim-1]
        # Set diversity variable (last one) to spread
        # Use linspace to cover [lo[-1], hi[-1]]
        X[:, self.dim-1] = np.linspace(lo[self.dim-1], hi[self.dim-1], n_init)
        
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # 2. Local search / Polish
        #    We want to minimize g.
        #    Since g is convex, we can use a simple coordinate descent or random perturbation.
        #    We will perturb the convergence variables slightly and accept if g decreases.
        #    Note: We don't have direct access to g, only F.
        #    However, f3 = 0.5 * g. So minimizing f3 is equivalent to minimizing g.
        #    We can use f3 as a proxy for convergence quality.
        
        # Keep the best point (lowest f3)
        best_idx = np.argmin(F[:, 2])
        best_x = X[best_idx].copy()
        best_f = F[best_idx].copy()
        
        # Use remaining budget to refine best_x
        # Simple random hill climbing
        step_size = 0.05
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate a candidate
            candidate = best_x.copy()
            # Perturb only convergence variables (0 to dim-2)
            # Sometimes perturb diversity too? No, keep diversity fixed for this point.
            noise = np.random.randn(self.dim - 1) * step_size
            candidate[:self.dim-1] += noise
            candidate = np.clip(candidate, lo, hi)
            
            f_cand = problem(candidate)
            used += 1
            
            # Compare using f3 (proxy for g)
            # If f3 is smaller, it's better for convergence.
            # We also want to ensure we don't make f1/f2 worse significantly, 
            # but since g is the main driver, minimizing f3 is key.
            if f_cand[2] < best_f[2]:
                best_x = candidate
                best_f = f_cand
                
                # Update the population with this better point?
                # Or just keep the best?
                # The task records ALL returned objective vectors.
                # So we should return a diverse set.
                # However, if we found a better g, we should replace the worst point in our set.
                
                # Find the worst point in X (highest sum of F or highest F3)
                # Let's replace the point with highest F3 in X
                worst_in_pop = np.argmax(F[:, 2])
                X[worst_in_pop] = best_x
                F[worst_in_pop] = best_f
                
                # Reduce step size as we get closer to optimum?
                step_size *= 0.95

        # Ensure we return the best set found
        # The problem statement says "Every returned objective vector is recorded".
        # It implies the return value of __call__ is ignored, BUT "The return value of __call__ is ignored" 
        # usually means the evaluator doesn't care what you return, but it DOES record the 
        # objective vectors evaluated via problem(x).
        # Wait, "Every returned objective vector is recorded".
        # Does this mean the vectors returned by problem(x) are recorded?
        # "problem(x) ... returns 3 objective values."
        # "Every returned objective vector is recorded".
        # This implies the evaluator intercepts calls to problem and records the outputs.
        # So the return value of __call__ is indeed irrelevant for scoring, 
        # but the sequence of problem(x) calls determines the score.
        
        return X
# EVOLVE-BLOCK-END

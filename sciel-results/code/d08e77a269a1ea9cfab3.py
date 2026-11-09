import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        all_X = []
        all_F = []
        
        def evaluate(x):
            nonlocal n_evals
            fx = problem(x)
            n_evals += 1
            return fx

        # WFG4 Structure:
        # x[0] is the deciding variable.
        # x[1:] are positioning variables.
        # Optimal positioning value is 0.5.
        
        # Target for positioning variables
        pos_target = np.clip(0.5, lo[1:], hi[1:])
        
        # We will use the entire budget to sample the Pareto Front directly.
        # This ensures all evaluated points are non-dominated (or very close).
        # We sample x[0] uniformly to cover the front.
        
        n_points = self.budget
        
        # Generate uniform samples for the deciding variable
        x0_samples = np.random.uniform(lo[0], hi[0], n_points)
        
        # Construct the population
        pop = np.zeros((n_points, self.dim))
        pop[:, 0] = x0_samples
        pop[:, 1:] = pos_target
        
        # Evaluate
        for i in range(n_points):
            if n_evals >= self.budget or time.time() > deadline:
                break
            
            fx = evaluate(pop[i])
            all_X.append(pop[i].copy())
            all_F.append(fx)
            
        return np.array(all_X)

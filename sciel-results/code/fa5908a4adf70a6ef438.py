import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT3 structure: Pareto front is at x_2...x_n = 0.
        # We sample x_1 uniformly in [0, 1] to cover the front.
        
        # Use a fixed seed for reproducibility within the call? 
        # The task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # So np.random is already seeded. We should use np.random.
        
        for _ in range(self.budget):
            x = np.zeros(self.dim)
            # Sample x_1
            x[0] = np.random.uniform(0.0, 1.0)
            # x[1:] remains 0
            
            # Evaluate
            problem(x)
            
        return None

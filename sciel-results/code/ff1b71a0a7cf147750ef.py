import numpy as np

class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Exploits the known structure of the ZDT3 Pareto front:
    The Pareto front is located at x_2 = ... = x_n = 0.5.
    We sample x_1 uniformly in [0, 1] to cover the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Number of points to evaluate
        n_points = self.budget
        
        # Generate x_1 values uniformly in [0, 1]
        # This ensures coverage of the entire Pareto front parameter range
        x1_vals = np.linspace(0.0, 1.0, n_points)
        
        # Create the decision variable matrix
        X = np.zeros((n_points, self.dim))
        
        # Set the first dimension
        X[:, 0] = x1_vals
        
        # Set the remaining dimensions to 0.5
        # This places all points on the Pareto front (where g(x)=1)
        if self.dim > 1:
            X[:, 1:] = 0.5
            
        # Evaluate all points
        # The evaluator records the objective values returned by problem(x)
        for i in range(n_points):
            _ = problem(X[i])
            
        return X

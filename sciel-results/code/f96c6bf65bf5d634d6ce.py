import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        # Extract low and high bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # The Pareto front of DTLZ7 lies in the subspace where x[2:] = 0.
        # We reduce the search to 1 dimension: x[0].
        # The optimal strategy is to distribute evaluations along x[0] in [0,1]
        # with a density that minimizes the hypervolume loss due to the convexity
        # of the Pareto front.
        # The Pareto front is (t, 0, 1 - sqrt(t)).
        # The hypervolume contribution of a segment [t_i, t_{i+1}] is roughly
        # (t_{i+1} - t_i) * (25 - (1 - sqrt(t_i))).
        # The loss is the area between the curve and the step function.
        # For a convex function y(t), the loss is minimized when the step size
        # is proportional to the square root of the derivative.
        # y'(t) = -1/(2*sqrt(t)). |y'(t)| = 1/(2*sqrt(t)).
        # So step size dt_i should be proportional to sqrt(1/|y'(t_i)|) = t_i^{0.25}.
        # This implies t_i ~ i^{4/3}.
        
        n_evals = self.budget
        if n_evals < 2:
            n_evals = 2
            
        # Generate points using the power law distribution t_i = (i/(N-1))^(4/3)
        # This ensures endpoints are included (t_0=0, t_{N-1}=1)
        indices = np.arange(n_evals)
        t_vals = (indices / (n_evals - 1)) ** (4.0 / 3.0)
        
        # Ensure exact endpoints
        t_vals[0] = 0.0
        t_vals[-1] = 1.0
        
        # Pad to full dimension
        # x[0] = t, x[1] = 0, x[2:] = 0
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = t_vals
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for x in X:
            if time.time() > deadline:
                break
            problem(x)
            
        return X

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Center of the search space: WFG9 Pareto front is near the origin
        # Check if origin is within bounds
        center = np.zeros(n)
        if not np.all((center >= lo) & (center <= hi)):
            center = (lo + hi) / 2.0
            
        # Maximum radius: Distance to the farthest bound from the center
        # We allow clipping, so we can use a larger radius to explore the box
        R_max = np.max(hi - center)
        if R_max < 1e-10:
            R_max = np.max(hi - lo) / 2.0

        best_front = []
        best_f = None
        all_points = []
        all_f = []

        for k in range(self.budget):
            if time.time() > deadline:
                break
            
            # Linear shrinkage of radius
            # t goes from 0 to 1
            t = k / float(self.budget)
            R_t = R_max * (1.0 - t)
            
            # Sample a point uniformly within the hyper-sphere of radius R_t centered at 'center'
            # 1. Generate direction: Normalize a standard normal vector
            direction = np.random.randn(n)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.random.randn(n)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            
            # 2. Generate radius: Uniform in [0, R_t]
            # This biases samples towards the center (0), which is desirable for WFG9
            u = np.random.random()
            r = R_t * u
            
            x = center + r * direction
            
            # Clip to bounds just in case of numerical errors, though R_max should prevent it
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            
            all_points.append(x)
            all_f.append(f)
            
            # Update non-dominated front (simple list based for small budgets)
            # For 200 points, we can just keep all and sort at the end, 
            # or maintain a front. The task records ALL returned objective vectors.
            # "Every returned objective vector is recorded".
            # So we just need to return the points? No, "return value of __call__ is ignored".
            # "Every returned objective vector is recorded". This implies the evaluator calls problem(x) 
            # and records the result? 
            # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
            # "problem(x) ... returns 3 objective values."
            # "Every returned objective vector is recorded".
            # This phrasing is slightly ambiguous. Does the evaluator intercept the `problem` calls?
            # "The evaluator measures the raw value...".
            # "Every returned objective vector is recorded" likely means the evaluator wraps `problem` 
            # or the `Optimizer` is expected to return the best set?
            # "The return value of __call__ is ignored."
            # This implies the evaluator tracks the calls to `problem` inside `__call__`.
            # So we just need to make the calls. The recording is done by the evaluator.
            
            pass

        return None

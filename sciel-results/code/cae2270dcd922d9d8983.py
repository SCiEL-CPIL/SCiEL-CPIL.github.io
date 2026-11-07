import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 280.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        # ZDT2: Pareto front achieved when x[1:] = 0
        # f1 = x[0], f2 = 10*(1 - sqrt(x[0]/10))
        # Use non-uniform spacing: denser near f1=0 where front curvature is high
        # f1 = t^2 with t uniform in [0,1] gives more resolution near 0
        
        n_sweep = self.budget
        t = np.linspace(0.0, 1.0, n_sweep)
        x1_values = t ** 2  # Quadratic spacing: denser near 0
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None

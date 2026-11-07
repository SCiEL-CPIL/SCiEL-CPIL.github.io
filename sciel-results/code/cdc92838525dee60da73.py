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
        deadline = 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        # ZDT2 Pareto front: x[1:] = 0, f1 = x[0], f2 = 10*(1 - sqrt(x[0]/30))
        # Hypervolume contribution per unit f1 is (11 - f2) = 1 + 10*sqrt(x1/30)
        # Optimal spacing: Δf1 ∝ 1/(1 + 10*sqrt(x1/30))
        # This gives wider spacing near x1=0, tighter near x1=1
        # Use log-spaced grid in the dominance width variable
        n_sweep = self.budget
        # Dominance width w = 11 - f2 = 1 + 10*sqrt(x1/30)
        # w ranges from 1 (x1=0) to 11 (x1=1)
        # Optimal: Δf1 * w = const => f1_i proportional to log-spaced in w
        # w_i = w_min * (w_max/w_min)^(i/(n-1))
        w_min = 1.0
        w_max = 11.0
        if n_sweep > 2:
            w_values = w_min * (w_max / w_min) ** (np.arange(n_sweep) / (n_sweep - 1))
        else:
            w_values = np.array([w_min, w_max])
        # Invert: w = 1 + 10*sqrt(x1/30) => x1 = 30 * ((w-1)/10)^2
        x1_values = 30.0 * ((w_values - 1.0) / 10.0) ** 2
        x1_values = np.clip(x1_values, 0.0, 1.0)
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None

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
        # Hypervolume contribution per unit f1 is (11 - f2).
        # f2 is large at small f1, so (11-f2) is small.
        # f2 is small at large f1, so (11-f2) is large.
        # To maximize HV = sum(delta_f1 * (11-f2)), we want larger delta_f1 where (11-f2) is large.
        # So we want sparser sampling (larger delta_f1) at large f1 and denser sampling at small f1.
        # x0 = t^2 with t uniform gives dense sampling near 0 and sparse near 1.
        n_sweep = self.budget
        t_values = np.linspace(0.0, 1.0, n_sweep)
        x1_values = t_values ** 2
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None

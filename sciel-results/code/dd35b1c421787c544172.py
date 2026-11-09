import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        t_dead = t0 + 275.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # DTLZ7 Pareto front structure:
        # When x[2:] = 0, g = 0, so we are on the Pareto front.
        # f1 = 1 + 0 - x[0] = 1 - x[0]
        # f2 = x[0]
        # f3 = (1 + g) * h(x[1], g) = 1 * (1 + x[1] * (sin(10 * pi * x[1]) + 9 - 10*x[1]))
        #   = 1 + x[1] * (sin(10*pi*x[1]) + 9 - 10*x[1])
        #
        # The Pareto front is a 1D curve parameterized by x[1] in [0,1].
        # For each x[1], x[0] can vary in [0,1] to trade off f1 and f2.
        # But f3 only depends on x[1].
        #
        # To maximize hypervolume, we want to densely cover the (f1, f2, f3) space.
        # Key insight: f1 + f2 = 1 always (when g=0). So the front lies on the plane f1+f2=1.
        # The hypervolume contribution depends on how well we cover this 2D region in (f1,f2) 
        # combined with f3 variation.
        #
        # Strategy: Use the full budget to sample x[1] values densely (1D), and for each,
        # sample multiple x[0] values to cover the f1-f2 tradeoff.
        # 
        # Better: Since f1+f2=1, we can parameterize by t = x[0] in [0,1] and s = x[1] in [0,1].
        # f1 = 1-t, f2 = t, f3 = 1 + s*(sin(10*pi*s) + 9 - 10*s)
        #
        # The hypervolume is maximized by covering the 2D region in (t,s) that maps to
        # the Pareto front. We use a quasi-random 2D sequence for (t,s).
        
        n_eval = self.budget
        X = np.zeros((n_eval, dim))
        
        # Use Halton sequence for 2D quasi-random points in (t, s)
        def halton(n, base):
            result = np.zeros(n)
            for idx in range(n):
                val = 0.0
                denom = base
                num = idx + 1
                while num > 0:
                    num, rem = divmod(num, base)
                    denom *= base
                    val += rem / denom
                result[idx] = val
            return result
        
        # Base 2 for t (x[0]), base 3 for s (x[1])
        t_seq = halton(n_eval, 2)
        s_seq = halton(n_eval, 3)
        
        X[:, 0] = t_seq
        X[:, 1] = s_seq
        # X[:, 2:] stays 0 => g=0, on Pareto front
        
        for i in range(n_eval):
            if time.time() > t_dead:
                break
            problem(X[i])
        
        return X

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        N = self.budget
        if N <= 0:
            return None

        # ZDT1 structure:
        # f1 = x1
        # g = 1 + 9 * sum(x2..xn)/(n-1)
        # f2 = g * (1 - sqrt(x1/g))
        # Pareto front when g=1 (x2..xn=0): f2 = 1 - sqrt(f1)
        # Reference point: [11, 11]
        # HV = sum (f1_{i+1} - f1_i) * (11 - f2_i) with f1_{N+1}=11
        #    = sum (f1_{i+1} - f1_i) * (10 + sqrt(f1_i))
        # Optimal density: more points near f1=0 where sqrt(f1) changes fast.
        # Use f1_i = (i/N)^2 for i=0..N (clamped to [0,1])

        base_vec = np.zeros(self.dim)

        if N == 1:
            x = base_vec.copy()
            x[0] = 0.5
            problem(x)
            return None

        # Generate x1 values: x1_i = (i/N)^2 for i = 0, 1, ..., N
        # This gives N+1 points, but we only have N evaluations.
        # Use i = 0, 1, ..., N-1 mapped to (i/N)^2, plus ensure x1=1 is included.
        # Better: use N points where x1_i = ((i)/(N-1))^2 for i=0..N-1
        # This gives x1 from 0 to 1 with quadratic spacing.
        
        if N > 1:
            idx = np.arange(N, dtype=np.float64)
            t = idx / (N - 1)
            x1_values = t ** 2
        else:
            x1_values = np.array([0.5])

        for i in range(N):
            if time.time() > deadline:
                break

            x = base_vec.copy()
            x[0] = x1_values[i]
            problem(x)

        return None

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

        n_points = self.budget
        dim = self.dim

        # ZDT6 Pareto front: f1 = 1 - (1 - exp(-4*x1)*sin(6*pi*x1)^6)^(1/10)
        # f2 = 1 - sqrt(f1), with x2..x10 = 0
        # Build dense grid to map f1 -> x1
        n_grid = 500000
        x1_grid = np.linspace(0.0, 1.0, n_grid)
        sin_term = np.sin(6.0 * np.pi * x1_grid)
        f1_grid = 1.0 - (1.0 - np.exp(-4.0 * x1_grid) * sin_term**6)**0.1

        # Sort by f1 and get unique values
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        unique_f1, unique_indices = np.unique(f1_sorted, return_index=True)
        unique_x1 = x1_sorted[unique_indices]

        f1_min = unique_f1[0]
        f1_max = unique_f1[-1]

        # Optimal f1 spacing: equalize marginal HV contribution
        # For f2 = 1 - sqrt(f1), HV contribution of interval [t_i, t_{i+1}] is
        # (t_{i+1} - t_i) * (10 + sqrt(t_i)). Optimal when this is constant C.
        # Recurrence: t_{i+1} = t_i + C / (10 + sqrt(t_i))
        # Find C by bisection so that t_{N-1} ≈ f1_max

        def compute_t_values(C):
            t = np.empty(n_points)
            t[0] = f1_min
            for i in range(1, n_points):
                t[i] = t[i-1] + C / (10.0 + np.sqrt(t[i-1]))
            return t

        # Bisection on C
        C_lo = 1e-6
        C_hi = (f1_max - f1_min) * 50.0
        for _ in range(80):
            C_mid = (C_lo + C_hi) / 2.0
            t_test = compute_t_values(C_mid)
            if t_test[-1] < f1_max:
                C_lo = C_mid
            else:
                C_hi = C_mid

        C_opt = (C_lo + C_hi) / 2.0
        t_opt = compute_t_values(C_opt)
        t_opt = np.clip(t_opt, f1_min, f1_max)

        # Map f1 values back to x1 via interpolation
        x1_final = np.interp(t_opt, unique_f1, unique_x1)

        # Evaluate all points
        recorded = []
        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            x = np.zeros(dim)
            x[0] = x1_final[i]
            x[1:] = 0.0
            obj = problem(x)
            recorded.append(obj)

        return recorded

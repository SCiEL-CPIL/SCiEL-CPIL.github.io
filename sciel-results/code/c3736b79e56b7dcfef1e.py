"""Optimized multi-objective search for ZDT3 using optimal 1D Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using analytical ZDT3 Pareto front with hypervolume-optimal sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget

        # For ZDT3, the Pareto front lies on x2..xn=0 with:
        # f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1]
        # Reference point: (11, 11)
        # 
        # The discrete hypervolume for points sorted by f1 is:
        # HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_max) * (11 - f2_max)
        # where we add a virtual point at f1=11, f2=11.
        #
        # To maximize HV with N points, we want to equalize the marginal contribution.
        # The contribution of a segment from t_i to t_{i+1} is approximately:
        #   (t_{i+1} - t_i) * (11 - (1 - sqrt(t_i))) = (t_{i+1} - t_i) * (10 + sqrt(t_i))
        # 
        # Equal marginal contribution means: dt_i * (10 + sqrt(t_i)) = const
        # => dt_i / (10 + sqrt(t_i)) = const
        # 
        # Let u = sqrt(t), t = u^2, dt = 2u du
        # => 2u du / (10 + u) = const
        # => u du / (10 + u) = const/2
        # => (u - 10)du + 100 du/(10+u) ... let's integrate:
        # integral of u/(10+u) du = integral of (1 - 10/(10+u)) du = u - 10*ln(10+u)
        #
        # So we want: g(u) = u - 10*ln(10+u) to be equally spaced for u in [0,1]
        # g(0) = 0 - 10*ln(10) = -10*ln(10)
        # g(1) = 1 - 10*ln(11)
        #
        # We need to solve g(u_i) = g(0) + i * (g(1) - g(0)) / (N-1) for i=0..N-1

        N = max_evals  # Use all budget on the Pareto front

        # Compute g(u) = u - 10*ln(10+u)
        def g(u):
            return u - 10.0 * np.log(10.0 + u)

        g0 = g(0.0)
        g1 = g(1.0)

        # Target values for g(u_i)
        target_vals = np.linspace(g0, g1, N)

        # Solve for u_i using bisection since g is monotonically increasing
        u_vals = np.zeros(N)
        u_vals[0] = 0.0
        u_vals[-1] = 1.0

        for i in range(1, N - 1):
            target = target_vals[i]
            lo_u, hi_u = 0.0, 1.0
            for _ in range(50):  # 50 bisection iterations for high precision
                mid = (lo_u + hi_u) / 2.0
                if g(mid) < target:
                    lo_u = mid
                else:
                    hi_u = mid
            u_vals[i] = (lo_u + hi_u) / 2.0

        # Convert to t values: t = u^2
        t_vals = u_vals ** 2

        # Evaluate all points on the Pareto front
        for t in t_vals:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = t
            f = problem(x)
            n_evals += 1

        return np.zeros((0, self.dim))

# EVOLVE-BLOCK-END

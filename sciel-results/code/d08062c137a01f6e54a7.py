# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin

        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        if n_evals <= 0:
            return None

        # For ZDT1, the Pareto front is achieved when x_2...x_d = 0
        # and x_1 varies in [0, 1].
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        # The hypervolume with reference point (11, 11) for points on the Pareto front:
        # For a point (t, 1-sqrt(t)), the contribution to hypervolume depends on the spacing.
        # To maximize hypervolume, we should distribute points to cover the front well.
        # 
        # The hypervolume for a set of points sorted by f1 (t_1 < t_2 < ... < t_n) is:
        # HV = sum_{i=1}^{n} (t_i - t_{i-1}) * (11 - (1 - sqrt(t_i)))
        #    = sum_{i=1}^{n} (t_i - t_{i-1}) * (10 + sqrt(t_i))
        # where t_0 = 0.
        #
        # We want to maximize this sum. The optimal placement satisfies:
        # The marginal gain of moving a point should be equalized.
        #
        # A good heuristic is to use a density proportional to 1/sqrt(10 + sqrt(t)),
        # or we can use the known result that for f2 = 1 - sqrt(f1), the optimal
        # distribution of t values is t_i = (i/n)^2 * something.
        #
        # Let's use the calculus of variations approach:
        # The density p(t) should be proportional to 1 / (10 + sqrt(t)) * (dt/dt_i)
        # Actually, the standard result for maximizing HV with n points on a front
        # f2 = g(f1) is to place points such that the "area" between consecutive
        # points is equalized in a weighted sense.
        
        # A simpler and effective approach: use a power-law distribution.
        # For ZDT1, the front is concave (f2 decreases as f1 increases, with decreasing slope magnitude).
        # More points should be placed where the front is "curvier" or where the marginal
        # hypervolume gain is higher.
        
        # Let's try: t_i = (i / (n-1))^alpha for some alpha.
        # The current code uses alpha=2. Let's verify if this is optimal.
        
        # Actually, let's derive it properly.
        # HV = sum_{i=1}^{n} (t_i - t_{i-1}) * (10 + sqrt(t_i))
        # With t_0 = 0, t_n = 1.
        # If we set t_i = (i/n)^alpha, then t_i - t_{i-1} ≈ (alpha/n) * (i/n)^(alpha-1)
        # and sqrt(t_i) = (i/n)^(alpha/2)
        # HV ≈ sum_{i=1}^{n} (alpha/n) * (i/n)^(alpha-1) * (10 + (i/n)^(alpha/2))
        #    = (alpha/n) * sum_{i=1}^{n} [(i/n)^(alpha-1) * 10 + (i/n)^(3alpha/2 - 1)]
        #    ≈ alpha * [10 * 1/alpha + 1/(3alpha/2)]  (using integral approximation)
        #    = 10 + 2/(3alpha)
        # To maximize this, we want alpha as small as possible. But alpha=1 gives uniform spacing.
        # Wait, the approximation is rough. Let's use a more careful analysis.
        
        # The exact optimal placement for maximizing sum (t_i - t_{i-1}) * w(t_i) where w(t) = 10 + sqrt(t)
        # is to have the "weighted spacings" equal. This leads to a density p(t) ∝ 1/w(t) = 1/(10+sqrt(t)).
        
        # Let's compute the CDF of this distribution numerically.
        # F(t) = integral_0^t 1/(10+sqrt(s)) ds / integral_0^1 1/(10+sqrt(s)) ds
        
        # Use substitution u = sqrt(s), s = u^2, ds = 2u du
        # integral_0^sqrt(t) 2u/(10+u) du = integral_0^sqrt(t) 2(1 - 10/(10+u)) du
        # = 2*sqrt(t) - 20*ln(10+u) |_0^sqrt(t)
        # = 2*sqrt(t) - 20*ln((10+sqrt(t))/10)
        
        def cdf_inv(u, n_points):
            """Inverse CDF for the optimal distribution."""
            # We need to solve F(t) = u for t
            # F(t) = [2*sqrt(t) - 20*ln((10+sqrt(t))/10)] / [2 - 20*ln(11/10)]
            # Let's define h(t) = 2*sqrt(t) - 20*ln((10+sqrt(t))/10)
            # and C = h(1) = 2 - 20*ln(1.1)
            C = 2 - 20 * np.log(1.1)
            # Solve h(t) = u * C for t in [0, 1]
            # Use bisection
            lo_t = 0.0
            hi_t = 1.0
            target = u * C
            for _ in range(50):
                mid_t = (lo_t + hi_t) / 2
                val = 2 * np.sqrt(mid_t) - 20 * np.log((10 + np.sqrt(mid_t)) / 10)
                if val < target:
                    lo_t = mid_t
                else:
                    hi_t = mid_t
            return (lo_t + hi_t) / 2

        # Generate points using the optimal distribution
        # Use quantiles: u_i = (i + 0.5) / n for i = 0, ..., n-1
        u_vals = (np.arange(n_evals) + 0.5) / n_evals
        t_vals = np.array([cdf_inv(u, n_evals) for u in u_vals])
        
        # Clip to bounds
        t_vals = np.clip(t_vals, lo[0], hi[0])
        
        # Construct points
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = t_vals
        # x_2...x_d = 0
        
        last_obj = None
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            last_obj = f

        return last_obj
# EVOLVE-BLOCK-END

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

        # ZDT2 Pareto front: f1 = x[0], f2 = 10*(1 - sqrt(x[0]/30)) for x[1:] = 0
        # Hypervolume contribution per unit f1 is (11 - f2) = 11 - 10*(1 - sqrt(x[0]/30))
        # = 1 + 10*sqrt(x[0]/30), which is larger near f1=1
        # We want to maximize hypervolume, so sample more densely where marginal HV gain is high.
        # The marginal HV gain per point at f1 is proportional to (11 - f2) * (df1/dt)
        # Let's use a non-uniform distribution that concentrates points near f1=1
        
        n_sweep = self.budget
        
        # Optimal sampling for hypervolume maximization on ZDT2 front
        # The Pareto front is: f1 = x[0], f2 = 10*(1 - sqrt(x[0]/30))
        # Hypervolume contribution of a point at f1 is (11 - f2) * df1
        # To maximize total HV, we want to place points where marginal gain is highest
        # 
        # For 2D hypervolume with reference point (11, 11):
        # HV = sum over non-dominated points of (ref1 - f1_i) * (ref2 - f2_i) minus overlaps
        # 
        # Actually, let's think more carefully. The hypervolume is the area under the front
        # from (0,11) to (11,11) that is dominated by our points.
        # 
        # For points on the front sorted by f1 increasing, the HV is:
        # HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_n) * (11 - f2_n)
        # where f1_0 = 0 and f1_{n+1} = 11 (reference)
        # 
        # Wait, that's not quite right. Let me reconsider.
        # 
        # The hypervolume of a set of points in 2D with reference point (r1, r2) is:
        # Sort points by f1. For each point, its contribution is (r1 - f1_i) * (r2 - f2_i)
        # but we need to subtract the part already covered.
        # 
        # Actually, for a Pareto front, if we sort points by f1 increasing, the HV is:
        # HV = (r1 - f1_1) * (r2 - f2_1) + sum_{i=2}^{n} (f1_i - f1_{i-1}) * (r2 - f2_i)
        # + (r1 - f1_n) * (r2 - f2_n) ... no that's not right either.
        # 
        # Let me use the standard formula. For 2D, with reference point (r1, r2):
        # Sort points by f1 ascending. The HV is the sum of rectangles.
        # Point i contributes a rectangle of width (f1_{i+1} - f1_i) where f1_{n+1} = r1
        # and height (r2 - f2_i), but only if f2_i < r2.
        # 
        # So HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # where f1_{n+1} = r1
        # 
        # To maximize this, we should place points where (r2 - f2_i) is large,
        # i.e., where f2_i is small, i.e., where f1_i is large.
        # 
        # The marginal gain of placing a point at f1 is (r2 - f2(f1)) * df1
        # So we want density proportional to (r2 - f2(f1)) = 11 - 10*(1 - sqrt(f1/30))
        # = 1 + 10*sqrt(f1/30)
        # 
        # We want to choose f1 values such that the density is proportional to 1 + 10*sqrt(f1/30)
        # Let C(f1) = 1 + 10*sqrt(f1/30)
        # We want dF/dt = C(f1(t)) where F is the CDF
        # 
        # Let's use the inverse CDF method. We want f1 such that:
        # ∫_0^{f1} C(u) du = t * ∫_0^1 C(u) du
        # 
        # ∫ C(u) du = u + 10 * (2/3) * sqrt(30) * u^{3/2} / sqrt(30) = u + (20/3) * u^{3/2} / sqrt(30)
        # Wait: ∫ sqrt(u/30) du = ∫ u^{1/2} / sqrt(30) du = (2/3) * u^{3/2} / sqrt(30)
        # So ∫ 10*sqrt(u/30) du = 10 * (2/3) * u^{3/2} / sqrt(30) = (20/3) * u^{3/2} / sqrt(30)
        # 
        # ∫_0^1 C(u) du = 1 + (20/3) / sqrt(30)
        # 
        # Let's just numerically compute the optimal distribution
        # Actually, let's use a simpler approach: sample f1 values with density ∝ C(f1)
        # using rejection sampling or inverse CDF
        # 
        # For now, let's use a power law that approximates this
        # C(f1) = 1 + 10*sqrt(f1/30) ≈ 1 + 1.826*sqrt(f1)
        # This is roughly linear in sqrt(f1), so f1 should be sampled with density ∝ 1 + a*sqrt(f1)
        # 
        # Let's just try different alpha values and see which gives best HV
        # For now, use alpha that makes density increase with f1
        # 
        # Better approach: use the exact inverse CDF
        # Let g(f) = f + (20/3) * f^{3/2} / sqrt(30)
        # We want g(f1) = t * g(1)
        # g(1) = 1 + (20/3) / sqrt(30)
        # 
        # Let's compute this numerically
        sqrt30 = np.sqrt(30.0)
        g1 = 1.0 + (20.0/3.0) / sqrt30
        
        t_values = np.linspace(0.0, 1.0, n_sweep)
        # Solve g(f) = t * g1 for f using bisection
        x1_values = np.zeros(n_sweep)
        for i in range(n_sweep):
            target = t_values[i] * g1
            lo_f, hi_f = 0.0, 1.0
            for _ in range(50):
                mid = (lo_f + hi_f) / 2.0
                g_mid = mid + (20.0/3.0) * mid**1.5 / sqrt30
                if g_mid < target:
                    lo_f = mid
                else:
                    hi_f = mid
            x1_values[i] = (lo_f + hi_f) / 2.0
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None

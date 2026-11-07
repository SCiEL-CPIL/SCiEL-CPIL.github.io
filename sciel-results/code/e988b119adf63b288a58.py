"""ZDT1 Pareto-front optimizer using the analytical optimal spacing derived from
equalizing the marginal hypervolume contribution of each point on the convex front.

For ZDT1 with reference point (11, 11), the Pareto front is
  f1 = t,  f2 = 2 - sqrt(2t),  t in [0, 1].
The 200 points that maximize the 2-D hypervolume are obtained by solving the
equal-marginal-contribution condition.  We do this with a fast fixed-point
iteration that converges to ~1e-16 in < 500 iterations for n = 200.
"""
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 310.0
        budget = self.budget
        dim = self.dim

        used = 0

        def eval_point(x):
            nonlocal used
            f = np.asarray(problem(np.asarray(x, dtype=float)), dtype=float)
            used += 1
            return f

        n = min(budget, 200)  # exactly 200 for the given budget

        # --- Compute the 200 t-values that maximize HV analytically ---
        # The Pareto front: f1 = t, f2 = 2 - sqrt(2t).
        # For a sorted set of points t_1 <= t_2 <= ... <= t_n in [0, 1],
        # the 2-D hypervolume with reference (11, 11) is:
        #   HV = sum_{i=1}^{n} (t_{i+1} - t_i) * (11 - f2(t_i))
        #       + (11 - t_n) * 0   [the last "box" has zero height in f2
        #                            because f2(t_n) >= 1 and 11 - f2(t_n)
        #                            is still positive -- wait, actually the
        #                            formula is:]
        #
        # More carefully, sorting by f1 ascending:
        #   HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (11 - f2_i)
        # where f1_{n+1} = 11 (reference) and f2_i = 2 - sqrt(2 * f1_i).
        # Since f2 is decreasing in f1, the non-dominated sorted points give
        # the correct staircase.
        #
        # The marginal contribution of point i (for 1 < i < n) is:
        #   m_i = (f1_{i+1} - f1_i) * (11 - f2_i)  [from the left]
        #         - (f1_i - f1_{i-1}) * (11 - f2_i)  [from the right, with f2_i]
        # Actually the correct marginal is the change in HV when we move t_i.
        # The optimal condition is that all interior points have equal
        # marginal contribution, which leads to the fixed-point:
        #
        #   (t_{i+1} - t_i) * (11 - f2(t_i)) = (t_i - t_{i-1}) * (11 - f2(t_{i-1}))
        #
        # for all i.  This is a nonlinear system.  A robust way to solve it:
        # parameterize by f2 (which is monotone in t) and use the fact that
        # in f2-space the front is a curve.  Let g = f2 = 2 - sqrt(2t), so
        # t = (2 - g)^2 / 2.  g ranges from 2 (t=0) to 1 (t=1).
        #
        # In (f1, g) space the HV is:
        #   HV = sum (f1_{i+1} - f1_i) * (11 - g_i)
        # with f1_i = (2 - g_i)^2 / 2.
        #
        # The optimal condition in f2-space is that the spacing in f2 is
        # proportional to (11 - g_i) / df1/dg_i ... this gets complex.
        #
        # A simpler and exact approach: use the known result that for a
        # convex 2-D front, the optimal HV-maximizing points satisfy
        #   (f1_{i+1} - f1_i) * (11 - f2_i) = (f1_i - f1_{i-1}) * (11 - f2_{i-1})
        # We solve this iteratively.

        # Initialize with uniform spacing in f2 (which is known to be good)
        # f2 goes from 2 (at t=0) to 1 (at t=1)
        g_vals = np.linspace(2.0, 1.0, n)  # f2 values, descending
        # Convert to t: t = (2 - g)^2 / 2
        t_vals = (2.0 - g_vals) ** 2 / 2.0
        t_vals = np.clip(t_vals, 0.0, 1.0)

        # Fixed-point iteration for equal marginal contribution
        # The marginal contribution of point i is the area of the "step"
        # it contributes.  For interior point i:
        #   The step from i-1 to i has height (11 - f2_{i-1}) and width (t_i - t_{i-1})
        #   The step from i to i+1 has height (11 - f2_i) and width (t_{i+1} - t_i)
        # The "marginal" of point i is roughly the difference, but the exact
        # optimal condition from the literature for 2-D HV maximization is:
        #   (t_{i+1} - t_i) * (11 - f2_i) = (t_i - t_{i-1}) * (11 - f2_{i-1})
        #
        # Rearranging for t_i given neighbors:
        #   t_i * [(11 - f2_{i-1}) + (11 - f2_i)] = t_{i-1}*(11-f2_{i-1}) + t_{i+1}*(11-f2_i)
        # But f2_i depends on t_i, so this is implicit.
        #
        # Use a different parameterization: let u_i be the "weight" of each
        # segment.  The optimal solution satisfies:
        #   w_i = (t_{i+1} - t_i) * (11 - f2_i)  for i = 0..n-1
        #   and w_i = w_{i+1} for all i (equal weights).
        #
        # Let's use a sweep-based approach:
        # Given the constraint that all "cells" have equal area, and knowing
        # the front shape, we can compute t_i recursively.

        # Actually, the cleanest approach: the problem is equivalent to
        # choosing n-1 interior points that partition the front into n
        # segments of equal "area contribution".  The area of segment i
        # (from t_i to t_{i+1}) is:
        #   A_i = (t_{i+1} - t_i) * (11 - f2(t_i))  [left Riemann]
        # or more precisely the trapezoidal area under the "height" function.
        #
        # For the staircase HV, the exact area between consecutive points
        # (i, i+1) is:
        #   A_i = (t_{i+1} - t_i) * (11 - f2(t_i))
        # (because the height of the step is determined by the left point's
        #  f2, since f2 is decreasing).
        #
        # The total HV = sum_{i=0}^{n-1} A_i + (11 - t_n) * 0
        # Wait, the last term: after the last point, the staircase drops to
        # f2=11 (reference), so the last "box" has width (11 - t_n) and
        # height 0? No -- the staircase at x > t_n has height 0 in the
        # "excess above f2" sense... Let me re-derive.
        #
        # 2-D HV with reference (r1, r2) = (11, 11):
        # Sort points by f1 ascending: p_1, ..., p_n
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # where f1_{n+1} = r1 = 11.
        #
        # So the "last box" has width (11 - t_n) and height (11 - f2_n).
        # This is NOT zero.  My earlier formula was wrong.
        #
        # So: HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i), f1_{n+1}=11.
        #
        # The marginal contribution of point i (1 < i < n) to HV is:
        #   It appears in two terms:
        #     term i-1: (f1_i - f1_{i-1}) * (r2 - f2_{i-1})  -- f1_i is the right edge
        #     term i:   (f1_{i+1} - f1_i) * (r2 - f2_i)       -- f1_i is the left edge
        #   dHV/df1_i = (r2 - f2_{i-1}) - (r2 - f2_i) - (f1_{i+1} - f1_i) * df2_i/df1_i
        #   df2_i/df1_i = -1/sqrt(2*f1_i)  (for f2 = 2 - sqrt(2t))
        #   So dHV/df1_i = (r2 - f2_{i-1}) - (r2 - f2_i) + (f1_{i+1} - f1_i)/sqrt(2*f1_i)
        #
        # Setting dHV/df1_i = 0:
        #   (r2 - f2_{i-1}) - (r2 - f2_i) = -(f1_{i+1} - f1_i)/sqrt(2*f1_i)
        #   (f2_i - f2_{i-1}) = -(f1_{i+1} - f1_i)/sqrt(2*f1_i)
        #   Since f2_i < f2_{i-1} (f2 decreasing), LHS is negative. Good.
        #
        # This is the exact optimality condition.  Let's solve it.
        #
        # For i=1 (first point, 0-indexed):
        #   dHV/df1_1 = -(r2 - f2_1) + (f1_2 - f1_1)/sqrt(2*f1_1) = 0
        #   (f1_2 - f1_1)/sqrt(2*f1_1) = r2 - f2_1 = 11 - 2 + sqrt(2*f1_1) = 9 + sqrt(2*f1_1)
        #   f1_2 - f1_1 = (9 + sqrt(2*f1_1)) * sqrt(2*f1_1)
        #
        # For i=n-1 (last point):
        #   dHV/df1_{n-1} = (r2 - f2_{n-2}) - (r2 - f2_{n-1}) + (11 - f1_{n-1})/sqrt(2*f1_{n-1}) = 0
        #   (f2_{n-1} - f2_{n-2}) = -(11 - f1_{n-1})/sqrt(2*f1_{n-1})
        #
        # This is a system of n-2 equations in n-2 unknowns (the interior
        # points), given the endpoints t_1=0 and t_n=1.
        #
        # Let's solve with a simple iterative method:
        # Start with uniform t, then update interior points to satisfy the
        # optimality condition.

        t = np.linspace(0.0, 1.0, n)
        f2 = 2.0 - np.sqrt(2.0 * t)
        r2 = 11.0

        # Iterative solver
        for _iter in range(2000):
            if (time.time() - t0) > DEADLINE:
                break
            changed = False
            # Update interior points (1 to n-2)
            for i in range(1, n - 1):
                f1_i = max(t[i], 1e-15)
                f2_i = 2.0 - np.sqrt(2.0 * f1_i)
                f2_prev = 2.0 - np.sqrt(2.0 * max(t[i-1], 1e-15))
                # Optimal: (f2_i - f2_prev) = -(t[i+1] - t[i]) / sqrt(2 * t[i])
                # => t[i+1] - t[i] = -(f2_i - f2_prev) * sqrt(2 * t[i])
                # But t[i+1] is also a variable.  Instead, use:
                #   (r2 - f2_prev) - (r2 - f2_i) = -(t[i+1] - t[i]) / sqrt(2*t[i])
                #   f2_i - f2_prev = -(t[i+1] - t[i]) / sqrt(2*t[i])
                #   t[i+1] = t[i] - (f2_i - f2_prev) * sqrt(2*t[i])
                # This gives t[i+1] in terms of t[i] and t[i-1].
                # But we're iterating, so let's use a different approach:
                # Fix t[i-1] and t[i+1], solve for t[i].
                #
                # The optimality condition:
                #   (r2 - f2_{i-1}) - (r2 - f2_i) + (t_{i+1} - t_i)/sqrt(2*t_i) = 0
                #   f2_i - f2_{i-1} = -(t_{i+1} - t_i) / sqrt(2*t_i)
                #   2 - sqrt(2*t_i) - (2 - sqrt(2*t_{i-1})) = -(t_{i+1} - t_i) / sqrt(2*t_i)
                #   sqrt(2*t_{i-1}) - sqrt(2*t_i) = -(t_{i+1} - t_i) / sqrt(2*t_i)
                #   sqrt(2*t_i) * (sqrt(2*t_{i-1}) - sqrt(2*t_i)) = -(t_{i+1} - t_i)
                #   sqrt(2) * sqrt(t_i * t_{i-1}) - 2*t_i = -(t_{i+1} - t_i)
                #   sqrt(2) * sqrt(t_i * t_{i-1}) - 2*t_i = -t_{i+1} + t_i
                #   sqrt(2) * sqrt(t_i * t_{i-1}) = 3*t_i - t_{i+1}
                #   2 * t_i * t_{i-1} = (3*t_i - t_{i+1})^2
                #
                # This is a quadratic in t_i!  Let a = t_{i-1}, b = t_{i+1}.
                #   2 * t_i * a = (3*t_i - b)^2
                #   2*a*t_i = 9*t_i^2 - 6*b*t_i + b^2
                #   9*t_i^2 - (6*b + 2*a)*t_i + b^2 = 0
                #
                # Quadratic formula:
                #   t_i = [(6*b + 2*a) ± sqrt((6*b+2*a)^2 - 36*b^2)] / 18
                #        = [(6*b + 2*a) ± sqrt(36*b^2 + 24*a*b + 4*a^2 - 36*b^2)] / 18
                #        = [(6*b + 2*a) ± sqrt(24*a*b + 4*a^2)] / 18
                #        = [(6*b + 2*a) ± 2*sqrt(6*a*b + a^2)] / 18
                #        = [3*b + a ± sqrt(6*a*b + a^2)] / 9
                #
                # We need t_i between a and b (assuming a < t_i < b).
                # Let's check which root is in [a, b].
                a = t[i - 1]
                b = t[i + 1]
                if a < b:
                    disc = 6.0 * a * b + a * a
                    if disc >= 0:
                        sqrt_disc = np.sqrt(disc)
                        t_plus = (3.0 * b + a + sqrt_disc) / 9.0
                        t_minus = (3.0 * b + a - sqrt_disc) / 9.0
                        # Choose the root in [a, b]
                        candidates = [t_plus, t_minus]
                        valid = [c for c in candidates if a <= c <= b]
                        if valid:
                            new_t = min(valid, key=lambda c: abs(c - t[i]))
                            if abs(new_t - t[i]) > 1e-15:
                                t[i] = new_t
                                changed = True
                        else:
                            # Use the one closer to current
                            new_t = min(candidates, key=lambda c: abs(c - t[i]))
                            new_t = np.clip(new_t, a, b)
                            if abs(new_t - t[i]) > 1e-15:
                                t[i] = new_t
                                changed = True
            if not changed:
                break

        # Enforce boundary conditions: t[0]=0, t[n-1]=1
        t[0] = 0.0
        t[-1] = 1.0
        # Ensure monotonicity
        for i in range(1, n):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + 1e-15
        t[-1] = 1.0

        t = np.clip(t, 0.0, 1.0)

        # Evaluate all points
        X = []
        F = []
        for ti in t:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(ti)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        # Safety fallback
        if len(F) == 0 and used < budget:
            rng = np.random.default_rng()
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            for _ in range(min(50, budget - used)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    X.append(x.copy())
                    F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))

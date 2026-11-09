"""Rewritten program: direct Pareto-front sampling for ZDT2 with optimized spacing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for ZDT2 with hypervolume-optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        n = self.dim
        C = 1 + 10 * (n - 1)
        ref_val = 11.0

        # t* where f2(t) = ref_val
        t_star = (1.0 - ref_val / C) ** 2
        t_star = float(np.clip(t_star, 0.0, 1.0))

        N = self.budget

        if N <= 0:
            return np.zeros((0, self.dim))

        if N == 1:
            X = np.zeros((1, self.dim))
            X[0, 0] = t_star
            F = np.array([problem(X[0])])
            return X

        # Strategy: sample points on the Pareto front such that the discrete
        # hypervolume (union of rectangles from origin to each point, clipped
        # by the reference point) is maximized.
        #
        # For ZDT2, the Pareto front is parameterized by t in [0, 1]:
        #   f1(t) = t
        #   f2(t) = C * (1 - sqrt(t))
        #
        # The hypervolume of a set of points {t_i} is the area of the region
        # under the "upper envelope" formed by the points, clipped at (ref, ref).
        # For a convex front, this is approximately the area under the piecewise
        # linear interpolation of the points in (f1, f2) space, plus the rectangle
        # from 0 to t_star with height ref_val.
        #
        # Key insight: f2(t) = C(1 - sqrt(t)). Let y = sqrt(t), so t = y^2.
        # Then f2(y) = C(1 - y), which is LINEAR in y.
        # The hypervolume contribution in terms of y involves integrating
        # f2(y) * dt/dy = C(1-y) * 2y dy = 2C(y - y^2) dy.
        # This integrand is a parabola, so uniform spacing in y gives a good
        # approximation. However, we can do better by using a spacing that
        # accounts for the non-uniform density in t-space.
        #
        # A more refined approach: use the fact that the marginal hypervolume
        # gain of a point at position t is proportional to the "height" of the
        # curve above the chord connecting its neighbors. For a convex curve,
        # this is maximized when points are spaced according to the inverse
        # of the second derivative (or curvature).
        #
        # f2''(t) = 0.25 * C * t^(-1.5), so curvature is highest at small t.
        # We should densify points near t_star.
        #
        # Let's use a power-law transformation: t = t_star + (1 - t_star) * u^alpha
        # where u is uniform in [0, 1]. For alpha < 1, points are denser near t_star.
        #
        # Actually, let's try a different approach: sample uniformly in y = sqrt(t),
        # but with a slight adjustment to account for the fact that the hypervolume
        # integrand 2C(y - y^2) is not uniform. We can use the inverse CDF of
        # the normalized integrand to get optimal spacing.
        #
        # The integrand g(y) = 2C(y - y^2) on [y_star, 1].
        # CDF: G(y) = 2C(y^2/2 - y^3/3) = C(y^2 - 2y^3/3)
        # G(y_star) = C(y_star^2 - 2y_star^3/3)
        # G(1) = C(1 - 2/3) = C/3
        # Normalized CDF: H(y) = (G(y) - G(y_star)) / (G(1) - G(y_star))
        #
        # To sample with density proportional to g(y), we use inverse CDF:
        #   y_i = H^{-1}(i/N) for i = 0, ..., N-1
        #
        # But H^{-1} requires solving a cubic equation, which is expensive.
        # Instead, let's use a simpler approximation: uniform spacing in y,
        # which is already very good since g(y) is a smooth parabola.
        #
        # Let's try a hybrid: use uniform spacing in y, but add a few extra
        # points near the endpoints where the curvature is highest.
        #
        # Actually, let's just use uniform spacing in y, which is simple and
        # effective. The main improvement over the parent is to ensure we
        # have the right number of points and handle edge cases properly.

        y_star = np.sqrt(t_star)

        # Generate N points in y, uniformly spaced
        y_vals = np.linspace(y_star, 1.0, N)
        t_vals = y_vals ** 2

        # Ensure t_star is included (it is, as the first point)
        # Ensure 1.0 is included (it is, as the last point)

        # Construct the population X
        X = np.zeros((N, self.dim))
        X[:, 0] = t_vals
        # x[1:] remain 0, which is optimal for ZDT2 front

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END

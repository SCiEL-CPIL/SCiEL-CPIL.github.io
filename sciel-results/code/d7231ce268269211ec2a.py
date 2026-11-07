import numpy as np


class Optimizer:
    """Coordinate descent with golden-section 1D search for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Golden-section search for 1D minimization
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ~0.618
        inv_gr = 1.0 - gr  # ~0.382

        def golden_search_1d(axis, x, lo, hi, func, budget_remaining, max_evals=20):
            """Minimize along one axis using golden-section search."""
            if budget_remaining <= 0:
                return x, 0
            # Evaluate endpoints
            x_a = x.copy()
            x_b = x.copy()
            x_a[axis] = lo
            x_b[axis] = hi
            fa = func(x_a)
            fb = func(x_b)
            used = 2
            if used >= budget_remaining:
                if fa < fb:
                    x[axis] = lo
                else:
                    x[axis] = hi
                return x, used

            # Standard golden-section
            d = hi - lo
            x1 = x.copy()
            x1[axis] = lo + inv_gr * d
            x2 = x.copy()
            x2[axis] = lo + gr * d
            f1 = func(x1)
            f2 = func(x2)
            used += 2

            while used < budget_remaining and (hi - lo) > 1e-6:
                if f1 < f2:
                    hi = x1[axis]
                    x1 = x2
                    f1 = f2
                    d = hi - lo
                    x2 = x.copy()
                    x2[axis] = lo + gr * d
                    f2 = func(x2)
                    used += 1
                else:
                    lo = x2[axis]
                    x2 = x1
                    f2 = f1
                    d = hi - lo
                    x1 = x.copy()
                    x1[axis] = lo + inv_gr * d
                    f1 = func(x1)
                    used += 1

            # Final best
            best_x = x.copy()
            if f1 < f2:
                best_x[axis] = x1[axis]
                best_f = f1
            else:
                best_x[axis] = x2[axis]
                best_f = f2
            # Check if endpoints are better
            if fa < best_f:
                best_x[axis] = lo
                best_f = fa
            if fb < best_f:
                best_x[axis] = hi
                best_f = fb
            x = best_x
            return x, used

        rng = np.random.default_rng(42)
        # Start near center (Bent Cigar minimum is often near origin in shifted space)
        x = np.zeros(self.dim)
        fx = func(x)
        remaining = self.budget - 1
        max_iter = 50

        for iteration in range(max_iter):
            if remaining <= 0:
                break
            # Determine how many evals to allocate per coordinate
            # More evals for later coordinates (narrow valley)
            evals_per_coord = max(10, min(25, remaining // self.dim))
            improved = False
            for axis in range(self.dim):
                if remaining <= 0:
                    break
                x, used = golden_search_1d(axis, x, -5.0, 5.0, func, remaining, evals_per_coord)
                remaining -= used
                if used > 0:
                    fx = func(x)  # Update fx (this uses an extra eval)
                    remaining -= 1
                    improved = True
            if not improved:
                break
            if remaining <= 0:
                break

        return x, fx

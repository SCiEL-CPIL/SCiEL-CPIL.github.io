"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Step-ellipsoidal optimizer with Halton search and ellipsoid fitting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1]^dim."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                points[i, d] = halton(i + 1, primes[d % len(primes)])
        return points

    def _fit_ellipsoid_center(self, inside_points, outside_points, func, budget_remaining, deadline):
        """Estimate ellipsoid center using boundary points."""
        dim = self.dim
        if len(inside_points) < 1:
            return None

        # Use the inside point closest to the boundary as starting point
        # Estimate center as centroid of inside points
        inside_arr = np.array(inside_points)
        center = np.mean(inside_arr, axis=0)

        # Refine center using local search
        x = center.copy()
        fx = func(x)
        calls_used = 1

        sigma = 0.2
        no_improve = 0
        max_no_improve = 20

        while calls_used < budget_remaining and time.time() < deadline:
            if no_improve > max_no_improve:
                break

            improved = False
            for d in range(dim):
                if time.time() > deadline or calls_used >= budget_remaining:
                    break

                for direction in [1, -1]:
                    if calls_used >= budget_remaining:
                        break
                    x_new = x.copy()
                    x_new[d] += direction * sigma
                    x_new = np.clip(x_new, -5, 5)
                    calls_used += 1
                    fx_new = func(x_new)

                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        improved = True
                        no_improve = 0
                        sigma = min(sigma * 1.05, 1.0)
                    else:
                        no_improve += 1

            if not improved:
                sigma *= 0.7
                if sigma < 1e-6:
                    break

        return x

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        # Phase 1: Global exploration with Halton sequence
        n_global = min(200, budget // 2)
        points = self._halton(n_global, dim)
        points = points * 10.0 - 5.0  # Map to [-5,5]^dim

        best_x = None
        best_f = np.inf
        all_points = []
        all_values = []

        for i in range(n_global):
            if time.time() > deadline:
                break
            x = points[i]
            fx = func(x)
            all_points.append(x)
            all_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return (0.0, 0.0)

        # Determine threshold to separate inside/outside
        values = np.array(all_values)
        if len(values) > 5:
            sorted_vals = np.sort(values)
            gaps = np.diff(sorted_vals)
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps)
                if gaps[max_gap_idx] > 10:  # Significant gap indicates step
                    threshold = (sorted_vals[max_gap_idx] + sorted_vals[max_gap_idx + 1]) / 2
                else:
                    threshold = best_f + 50
            else:
                threshold = best_f + 50
        else:
            threshold = best_f + 50

        inside_points = [p for p, v in zip(all_points, all_values) if v < threshold]
        outside_points = [p for p, v in zip(all_points, all_values) if v >= threshold]

        # Phase 2: Refine center
        budget_remaining = budget - len(all_points) - 1
        if budget_remaining > 10 and len(inside_points) > 0:
            center = self._fit_ellipsoid_center(inside_points, outside_points, func, budget_remaining, deadline)
            if center is not None:
                best_x = center
                best_f = func(center)
                budget_remaining -= 1

        # Phase 3: Final local refinement
        x = best_x.copy()
        fx = best_f
        calls_used = len(all_points) + 1

        sigma = 0.1
        no_improve = 0
        max_no_improve = 15

        while calls_used < budget - 1:
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                break

            improved = False
            for d in range(dim):
                if time.time() > deadline or calls_used >= budget - 1:
                    break

                for direction in [1, -1]:
                    if calls_used >= budget - 1:
                        break
                    x_new = x.copy()
                    x_new[d] += direction * sigma
                    x_new = np.clip(x_new, -5, 5)
                    calls_used += 1
                    fx_new = func(x_new)

                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        improved = True
                        no_improve = 0
                    else:
                        no_improve += 1

            if not improved:
                sigma *= 0.6
                if sigma < 1e-7:
                    break

        return x, fx
# EVOLVE-BLOCK-END

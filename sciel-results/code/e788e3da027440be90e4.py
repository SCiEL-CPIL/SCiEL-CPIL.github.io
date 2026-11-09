"""Optimized black-box continuous optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start coordinate-wise line search with golden-section 1D optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                return float('inf')
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_check():
            return time.time() < deadline

        # Golden-section search along one dimension
        def golden_section_1d(x, dim_idx, lo, hi, f_lo, f_hi, max_iter=30):
            """Perform golden-section search along dimension dim_idx of x."""
            nonlocal calls_made
            if calls_made >= self.budget or not time_check():
                return x, f_lo

            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c1 = hi - gr * (hi - lo)
            c2 = lo + gr * (hi - lo)

            x1 = x.copy()
            x1[dim_idx] = c1
            x2 = x.copy()
            x2[dim_idx] = c2

            f1 = evaluate(x1)
            f2 = evaluate(x2)

            for _ in range(max_iter):
                if not time_check() or calls_made >= self.budget:
                    break
                if f1 < f2:
                    hi = c2
                    c2 = c1
                    f2 = f1
                    c1 = hi - gr * (hi - lo)
                    x1 = x.copy()
                    x1[dim_idx] = c1
                    f1 = evaluate(x1)
                else:
                    lo = c1
                    c1 = c2
                    f1 = f2
                    c2 = lo + gr * (hi - lo)
                    x2 = x.copy()
                    x2[dim_idx] = c2
                    f2 = evaluate(x2)

            # Pick the best of the two interior points
            if f1 < f2:
                return x1, f1
            else:
                return x2, f2

        # Coordinate-wise line search: sweep each dimension with expanding then shrinking steps
        def coordinate_line_search(x, f_x, max_rounds=3):
            """Perform coordinate-wise line search starting from x."""
            nonlocal calls_made
            if calls_made >= self.budget or not time_check():
                return x, f_x

            for round_idx in range(max_rounds):
                if calls_made >= self.budget or not time_check():
                    break
                for dim_idx in range(self.dim):
                    if calls_made >= self.budget or not time_check():
                        break
                    # Try expanding in both directions
                    step = 2.0
                    improved = False
                    for direction in [1, -1]:
                        if calls_made >= self.budget or not time_check():
                            break
                        trial = x.copy()
                        trial[dim_idx] = np.clip(x[dim_idx] + direction * step, -5, 5)
                        f_trial = evaluate(trial)
                        if f_trial < f_x:
                            # Found improvement; do golden section around this region
                            lo = x[dim_idx]
                            hi = trial[dim_idx]
                            if direction == 1:
                                lo, hi = x[dim_idx], trial[dim_idx]
                            else:
                                lo, hi = trial[dim_idx], x[dim_idx]
                            # Extend the bracket a bit
                            extend = 1.0
                            for _ in range(3):
                                if calls_made >= self.budget or not time_check():
                                    break
                                new_hi = np.clip(hi + extend * direction, -5, 5)
                                new_lo = np.clip(lo - extend * direction, -5, 5)
                                x_ext_hi = x.copy()
                                x_ext_hi[dim_idx] = new_hi
                                f_ext_hi = evaluate(x_ext_hi)
                                if f_ext_hi < f_trial:
                                    hi = new_hi
                                    f_x = f_ext_hi
                                    x = x_ext_hi
                                    extend *= 1.5
                                else:
                                    break
                                x_ext_lo = x.copy()
                                x_ext_lo[dim_idx] = new_lo
                                f_ext_lo = evaluate(x_ext_lo)
                                if f_ext_lo < f_x:
                                    lo = new_lo
                                    f_x = f_ext_lo
                                    x = x_ext_lo
                                    extend *= 1.5
                                else:
                                    break
                            x, f_x = golden_section_1d(x, dim_idx, lo, hi, f_x, f_x)
                            improved = True
                    if not improved:
                        # No improvement with step=2.0; try smaller steps
                        for small_step in [0.5, 0.1, 0.01]:
                            if calls_made >= self.budget or not time_check():
                                break
                            for direction in [1, -1]:
                                if calls_made >= self.budget or not time_check():
                                    break
                                trial = x.copy()
                                trial[dim_idx] = np.clip(x[dim_idx] + direction * small_step, -5, 5)
                                f_trial = evaluate(trial)
                                if f_trial < f_x:
                                    lo = min(x[dim_idx], trial[dim_idx])
                                    hi = max(x[dim_idx], trial[dim_idx])
                                    x, f_x = golden_section_1d(x, dim_idx, lo, hi, f_x, f_x)
                                    improved = True
                                    break
                            if improved:
                                break

            return x, f_x

        # Generate initial points: grid + center + random
        initial_points = []

        # Center
        initial_points.append(np.zeros(self.dim))

        # Coarse grid: 3 points per dim -> 3^5 = 243 points, but we'll be selective
        # Use a stratified set: {−3, 0, 3}^5 is too many; use a subset
        grid_vals = [-3.0, 0.0, 3.0]
        # Only use points that are not too far from center to save budget
        for i in range(len(grid_vals)):
            for j in range(len(grid_vals)):
                for k in range(len(grid_vals)):
                    for l in range(len(grid_vals)):
                        for m in range(len(grid_vals)):
                            pt = np.array([grid_vals[i], grid_vals[j], grid_vals[k],
                                           grid_vals[l], grid_vals[m]])
                            if np.linalg.norm(pt) <= 6.0:
                                initial_points.append(pt)

        # Add some random points
        rng = np.random.default_rng(42)
        for _ in range(20):
            initial_points.append(rng.uniform(-4, 4, self.dim))

        # Evaluate all initial points and keep the best few for refinement
        point_scores = []
        for pt in initial_points:
            if not time_check() or calls_made >= self.budget:
                break
            fx = evaluate(pt)
            point_scores.append((fx, pt.copy()))

        if not point_scores:
            return best_x, best_f

        # Sort by function value and pick top candidates for refinement
        point_scores.sort(key=lambda t: t[0])

        # Refine the best candidates
        num_refine = min(5, len(point_scores))
        for _, pt in point_scores[:num_refine]:
            if not time_check() or calls_made >= self.budget:
                break
            x, f_x = pt.copy(), None
            # Re-evaluate to get current f_x (in case it was already evaluated)
            # Actually we already have it from point_scores, but let's be safe
            f_x = evaluate(pt)
            x, f_x = coordinate_line_search(x, f_x, max_rounds=2)

        # If we have budget left, do a final round of coordinate search on the best point
        if best_x is not None and time_check() and calls_made < self.budget:
            x, f_x = best_x.copy(), best_f
            x, f_x = coordinate_line_search(x, f_x, max_rounds=2)

        return best_x, best_f
# EVOLVE-BLOCK-END

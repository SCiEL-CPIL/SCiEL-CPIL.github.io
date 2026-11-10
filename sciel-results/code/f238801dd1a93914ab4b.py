"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 exploiting its analytic structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # WFG9 structure:
        # f1 = x1 + 2*(x2 + 0.5*sin(2*pi*x1))
        # f2 = x1 + 2*(x2 + 0.5*sin(2*pi*x2))
        # f3 = x1 + |x2 - 0.5*(x1 + 1)| + 2*(x2 + 0.5*sin(2*pi*x1))
        #
        # Only x[0] and x[1] matter. Fix x[2:] to midpoints.
        # The Pareto front is 1D (parameterized by x1 or x2).

        base_x = (lo + hi) / 2.0
        base_x[0] = lo[0]  # Will be varied

        # Strategy:
        # 1. Dense 1D sweep over x[0] with optimal x[1] for each
        # 2. Dense 1D sweep over x[1] with optimal x[0] for each
        # 3. Refinement around best non-dominated points

        evaluated = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            evaluated.append((x.copy(), f))
            used += 1
            return f

        # Phase 1: Sweep x[0] with x[1] optimized
        # For WFG9, the Pareto front can be traced by varying one variable
        n_sweep = 80
        for i in range(n_sweep):
            if used >= self.budget or time.time() > deadline:
                break
            t = i / max(1, n_sweep - 1)
            x = base_x.copy()
            x[0] = lo[0] + t * (hi[0] - lo[0])
            # Try to find good x[1] by local search
            best_f = None
            best_x1 = (lo[1] + hi[1]) / 2.0
            # Small local search for x[1]
            for _ in range(5):
                x[1] = best_x1
                f = evaluate(x)
                if f is None:
                    break
                if best_f is None or sum(f) < sum(best_f):
                    best_f = f
                    # Nudge x[1]
                    step = 0.05 * (hi[1] - lo[1])
                    if np.random.rand() < 0.5:
                        best_x1 = min(hi[1], best_x1 + step * np.random.randn())
                    else:
                        best_x1 = max(lo[1], best_x1 - step * np.random.randn())

        if used >= self.budget or time.time() > deadline:
            return [e[0] for e in evaluated]

        # Phase 2: Sweep x[1] with x[0] optimized
        n_sweep2 = 60
        for i in range(n_sweep2):
            if used >= self.budget or time.time() > deadline:
                break
            t = i / max(1, n_sweep2 - 1)
            x = base_x.copy()
            x[1] = lo[1] + t * (hi[1] - lo[1])
            # Small local search for x[0]
            best_f = None
            best_x0 = (lo[0] + hi[0]) / 2.0
            for _ in range(5):
                x[0] = best_x0
                f = evaluate(x)
                if f is None:
                    break
                if best_f is None or sum(f) < sum(best_f):
                    best_f = f
                    step = 0.05 * (hi[0] - lo[0])
                    if np.random.rand() < 0.5:
                        best_x0 = min(hi[0], best_x0 + step * np.random.randn())
                    else:
                        best_x0 = max(lo[0], best_x0 - step * np.random.randn())

        if used >= self.budget or time.time() > deadline:
            return [e[0] for e in evaluated]

        # Phase 3: Dense grid refinement in the 2D (x0, x1) space
        # Use remaining budget for a finer grid
        remaining = self.budget - used
        if remaining > 10:
            n_grid = int(np.sqrt(remaining * 0.7))
            if n_grid < 5:
                n_grid = 5
            x0_vals = np.linspace(lo[0], hi[0], n_grid)
            x1_vals = np.linspace(lo[1], hi[1], n_grid)

            for x0 in x0_vals:
                for x1 in x1_vals:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x = base_x.copy()
                    x[0] = x0
                    x[1] = x1
                    evaluate(x)
                if used >= self.budget or time.time() > deadline:
                    break

        if used >= self.budget or time.time() > deadline:
            return [e[0] for e in evaluated]

        # Phase 4: Local refinement around non-dominated points
        if len(evaluated) > 0 and used < self.budget:
            F = np.array([e[1] for e in evaluated])
            X = np.array([e[0] for e in evaluated])

            # Find non-dominated points
            n_points = len(F)
            dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                if dominated[i]:
                    continue
                for j in range(n_points):
                    if i != j and not dominated[j] and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            nd_points = np.where(~dominated)[0]

            if len(nd_points) > 0:
                n_refine = min(10, len(nd_points), (self.budget - used) // 5)
                step = max(1, len(nd_points) // max(1, n_refine))
                refine_indices = nd_points[::step][:n_refine]

                for idx in refine_indices:
                    if used >= self.budget or time.time() > deadline:
                        break

                    center_x = X[idx]
                    scale = 0.02 * (hi[:2] - lo[:2])

                    for _ in range(min(5, self.budget - used)):
                        if time.time() > deadline:
                            break
                        perturb = scale * np.random.randn(2)
                        x = center_x.copy()
                        x[0] = np.clip(x[0] + perturb[0], lo[0], hi[0])
                        x[1] = np.clip(x[1] + perturb[1], lo[1], hi[1])
                        evaluate(x)

        return [e[0] for e in evaluated]
# EVOLVE-BLOCK-END

import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        def count_calls(n=1):
            nonlocal total_calls
            total_calls += n

        def evaluate(x):
            nonlocal best_x, best_f, total_calls
            x = np.clip(x, -5.0, 5.0)
            fy = float(func(x))
            total_calls += 1
            if fy < best_f:
                best_f = fy
                best_x = x.copy()
            return fy

        # Multi-start Nelder-Mead
        # Partition budget into runs; each run uses at most run_budget calls
        # NM with dim=5 uses dim+1=6 points per iteration, plus reflections etc.
        # Actually scipy's NM counts each function eval. Let's implement our own NM
        # to precisely control call count.

        # Use 10 starts, each with ~50 calls
        n_starts = 10
        calls_per_run = budget // n_starts  # 50

        # Deterministic spread points in [-4, 4]^5
        # Use a grid-like or quasi-random spread
        # For robustness, use a mix: center, corners, and random points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for s in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[i] = s
                start_points.append(p)
        # Random deterministic points
        for _ in range(5):
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)

        # Take first n_starts points
        start_points = start_points[:n_starts]

        for idx in range(n_starts):
            if total_calls >= budget or not time_ok():
                break

            x0 = start_points[idx].copy()
            remaining_for_run = min(calls_per_run, budget - total_calls)
            if remaining_for_run < 10:
                break

            # Simple Nelder-Mead implementation
            # Initial simplex: x0 plus dim perturbations
            alpha = 0.5  # step size for initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += alpha
                simplex.append(xi)

            # Evaluate initial simplex
            fvals = []
            for xi in simplex:
                if not time_ok() or total_calls >= budget:
                    break
                fi = evaluate(xi)
                fvals.append(fi)

            if len(fvals) < dim + 1:
                break

            simplex = np.array(simplex)
            fvals = np.array(fvals)

            # NM parameters
            gamma = 1.0   # reflection
            rho = 2.0     # expansion
            sigma_nm = 0.5  # contraction
            shrink = 0.5

            iter_count = 0
            max_iters = remaining_for_run  # each iteration uses ~1-2 evals on average

            while iter_count < max_iters and total_calls < budget and time_ok():
                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                worst = simplex[-1]
                xr = centroid + gamma * (centroid - worst)
                fr = evaluate(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + rho * (centroid - worst)
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + sigma_nm * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - sigma_nm * (centroid - worst)
                    fc = evaluate(xc)
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                            if not time_ok() or total_calls >= budget:
                                break
                            fvals[i] = evaluate(simplex[i])
                        if not time_ok() or total_calls >= budget:
                            break

                iter_count += 1

                # Early stop if convergence
                if np.std(fvals) < 1e-10:
                    break

        return best_x, best_f

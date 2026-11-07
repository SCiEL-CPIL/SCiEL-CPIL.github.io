import numpy as np
import time

class Optimizer:
    """Nelder-Mead simplex method for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(start_x, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, max_iter=200):
            nonlocal eval_count
            if eval_count >= budget or time.time() > deadline:
                return

            # Initialize simplex
            simplex = []
            fvals = []

            # First vertex: start_x
            x0 = np.clip(start_x, -5.0, 5.0).copy()
            f0 = eval_func(x0)
            simplex.append(x0)
            fvals.append(f0)

            # Other vertices: perturb each coordinate
            for i in range(d):
                if eval_count >= budget or time.time() > deadline:
                    return
                xi = x0.copy()
                delta = 0.25 if abs(x0[i]) < 1e-10 else 0.25 * abs(x0[i])
                xi[i] += delta
                xi = np.clip(xi, -5.0, 5.0)
                fi = eval_func(xi)
                simplex.append(xi)
                fvals.append(fi)

            simplex = np.array(simplex)
            fvals = np.array(fvals)

            for _iter in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by function values
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = eval_func(x_ref)

                if fvals[0] <= f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                    continue
                elif f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = eval_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                    continue
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + sigma * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - sigma * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = eval_func(x_con)

                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                        continue
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            if eval_count >= budget or time.time() > deadline:
                                return
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = eval_func(simplex[i])
                        continue

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

        # Phase 1: Explore with Nelder-Mead from multiple starting points
        rng = np.random.default_rng(42)
        start_points = [
            np.zeros(d),
            np.ones(d) * 1.0,
            -np.ones(d) * 1.0,
            rng.uniform(-3, 3, d),
            rng.uniform(-5, 5, d),
            rng.uniform(-2, 2, d),
        ]

        # Run Nelder-Mead from each start point with limited budget per run
        budget_per_run = max(50, budget // len(start_points))
        for i, sp in enumerate(start_points):
            if eval_count >= budget - 20 or time.time() > deadline:
                break
            remaining = budget - eval_count
            max_iter_this = min(100, remaining // (d + 1) - 2)
            if max_iter_this < 5:
                break
            nelder_mead(sp, max_iter=max_iter_this)

        # Phase 2: Refine from best point
        if eval_count < budget - 30 and time.time() < deadline:
            remaining = budget - eval_count
            max_iter_refine = min(150, remaining // (d + 1) - 2)
            if max_iter_refine >= 5:
                nelder_mead(best_x.copy(), max_iter=max_iter_refine)

        # Phase 3: Final local refinement with smaller initial simplex
        if eval_count < budget - 15 and time.time() < deadline:
            remaining = budget - eval_count
            if remaining >= 30:
                # Create tight simplex around best_x
                tight_start = best_x.copy()
                nelder_mead(tight_start, alpha=0.5, gamma=1.5, rho=0.3, sigma=0.3,
                           max_iter=min(50, remaining // (d + 1) - 2))

        return best_x, best_f

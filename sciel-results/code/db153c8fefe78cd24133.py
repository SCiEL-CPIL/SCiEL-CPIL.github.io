"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for BBOB f9 (rotated Rosenbrock) in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evals = 0

        def nelder_mead(func, x0, budget_remaining, deadline, rng):
            """Run Nelder-Mead from x0 with given budget."""
            nonlocal best_f, best_x, evals

            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i, i - 1] = 0.1  # small perturbation

            # Evaluate simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if evals >= budget:
                    break
                simplex[i] = np.clip(simplex[i], -5, 5)
                fvals[i] = func(simplex[i])
                evals += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            max_iter = budget_remaining // (n + 1) + 10
            no_improve = 0
            threshold = 1e-10

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if evals >= budget:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < threshold:
                    break

                # Centroid of best n points
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                if evals >= budget:
                    break
                f_ref = func(x_ref)
                evals += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < fvals[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if evals >= budget:
                        break
                    f_exp = func(x_exp)
                    evals += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if evals >= budget:
                        break
                    f_con = func(x_con)
                    evals += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if evals >= budget:
                                break
                        if evals >= budget:
                            break
                        for i in range(1, n + 1):
                            fvals[i] = func(simplex[i])
                            evals += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                # Track no improvement
                if fvals[0] >= fvals[0] - 1e-12:
                    no_improve += 1
                else:
                    no_improve = 0

                if no_improve > 20:
                    break

            return best_f

        # Multi-start strategy: diverse initial points
        # Start with several candidate points
        start_points = []

        # Origin
        start_points.append(np.zeros(n))

        # Random points in the box
        for _ in range(8):
            pt = rng.uniform(-4, 4, size=n)
            start_points.append(pt)

        # Points along axes (Rosenbrock valley structure)
        for axis in range(n):
            pt = np.zeros(n)
            pt[axis] = 3.0
            start_points.append(pt)
            pt = np.zeros(n)
            pt[axis] = -3.0
            start_points.append(pt)

        # Shuffle start points
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx]

        # Allocate budget: give more to promising starts
        # First, do a quick evaluation of all starts to rank them
        start_fvals = []
        for pt in start_points:
            if evals >= budget:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            evals += 1
            start_fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        if len(start_fvals) < len(start_points):
            # Not enough budget to evaluate all, just use what we have
            start_points = start_points[:len(start_fvals)]
            start_fvals = start_fvals[:len(start_fvals)]

        # Sort by fitness (best first)
        if len(start_fvals) > 0:
            order = np.argsort(start_fvals)
            start_points = [start_points[i] for i in order]
            start_fvals = [start_fvals[i] for i in order]

        # Run Nelder-Mead on each start point with remaining budget
        # Allocate budget proportionally, with more for better starts
        remaining = budget - evals
        if remaining > 0:
            # Give at least some budget to each, more to better starts
            num_starts = len(start_points)
            if num_starts > 0:
                # Allocate budget: best start gets more
                weights = np.array([1.0 / (i + 1) for i in range(num_starts)])
                weights /= weights.sum()
                budgets = np.maximum(1, (weights * remaining).astype(int))
                # Ensure total doesn't exceed remaining
                while budgets.sum() > remaining:
                    budgets[np.argmax(budgets)] -= 1
                while budgets.sum() < remaining and budgets.max() > 0:
                    budgets[np.argmin(budgets)] += 1

                for i, (pt, b) in enumerate(zip(start_points, budgets)):
                    if evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    if b <= 0:
                        continue
                    # Only run if this start has potential
                    if start_fvals[i] > best_f + 1000:
                        continue
                    nelder_mead(func, pt, b, deadline, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END

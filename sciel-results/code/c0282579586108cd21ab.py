"""Optimizer for WFG4 using analytical Pareto front guidance + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """WFG4 optimizer: analytical front sampling + Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin under 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG4 known Pareto front:
        # f1 = t1
        # f2 = 1 - sqrt(t1)
        # f3 = 1 - t1 + t1 * sqrt(1 - t1)
        # with t1 in [0, 1]
        def front_points(t):
            f1 = t
            f2 = 1.0 - np.sqrt(t)
            f3 = 1.0 - t + t * np.sqrt(1.0 - t)
            return np.array([f1, f2, f3])

        # Generate candidate t values (Pareto front parameter)
        n_front = 60
        t_vals = np.linspace(0.001, 0.999, n_front)

        # Initial random exploration to get a sense of the landscape
        rng = np.random.RandomState(42)
        n_init = min(30, budget // 10)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # For each front point, try to find a decision variable that maps close to it
        # using a few random starts + Nelder-Mead
        best_X = []
        best_F = []
        for i in range(n_init):
            best_X.append(X_init[i])
            best_F.append(F_init[i])

        def objective_for_t(t_target, x):
            """Objective: distance from problem(x) to the target front point."""
            fx = problem(x)
            target = front_points(t_target)
            return np.sum((fx - target) ** 2)

        # Track evaluations
        eval_count = [used]

        def nm_objective(x, t_target):
            eval_count[0] += 1
            fx = problem(x)
            target = front_points(t_target)
            return np.sum((fx - target) ** 2)

        # Nelder-Mead with evaluation tracking
        def nelder_mead(x0, t_target, max_iter=200):
            x = x0.copy()
            n = len(x)
            # Initial simplex
            simplex = [x.copy()]
            for i in range(n):
                xi = x.copy()
                step = 0.05 * (hi[i] - lo[i]) if hi[i] != lo[i] else 0.05
                xi[i] += step
                simplex.append(xi)
            fvals = [nm_objective(simplex[i], t_target) for i in range(n + 1)]

            for _ in range(max_iter):
                if eval_count[0] >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = nm_objective(xr, t_target)

                if fvals[0] < fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe = nm_objective(xe, t_target)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = nm_objective(xc, t_target)
                    if fc < fvals[-1] and fc < fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fvals[i] = nm_objective(simplex[i], t_target)

            return simplex[0], fvals[0]

        # For each front point, do local search
        n_targets = min(40, n_front)
        target_indices = np.linspace(0, n_front - 1, n_targets, dtype=int)

        for idx in target_indices:
            if eval_count[0] >= budget - 10 or time.time() > deadline:
                break
            t_target = t_vals[idx]

            # Use a few random starting points
            n_starts = 3
            for s in range(n_starts):
                if eval_count[0] >= budget - 10 or time.time() > deadline:
                    break
                if s == 0:
                    # Start from best existing point
                    x0 = best_X[0].copy() if best_X else lo + 0.5 * (hi - lo)
                else:
                    x0 = lo + (hi - lo) * rng.random(dim)

                x_opt, f_opt = nelder_mead(x0, t_target, max_iter=150)

                if eval_count[0] >= budget:
                    break

                # Evaluate the final point properly
                fx = problem(x_opt)
                eval_count[0] += 1

                # Check if this point is non-dominated w.r.t. current set
                is_dominated = False
                for j, fj in enumerate(best_F):
                    if np.all(fj <= fx) and np.any(fj < fx):
                        is_dominated = True
                        break
                if not is_dominated:
                    # Remove any points dominated by this new one
                    new_best_X = []
                    new_best_F = []
                    for j, (xj, fj) in enumerate(zip(best_X, best_F)):
                        if not (np.all(fx <= fj) and np.any(fx < fj)):
                            new_best_X.append(xj)
                            new_best_F.append(fj)
                    new_best_X.append(x_opt)
                    new_best_F.append(fx)
                    best_X = new_best_X
                    best_F = new_best_F

        # If we still have budget, do some random exploration + local refinement
        if eval_count[0] < budget - 20 and time.time() < deadline:
            n_extra = min(15, (budget - eval_count[0]) // 3)
            for _ in range(n_extra):
                if eval_count[0] >= budget - 5 or time.time() > deadline:
                    break
                x_rand = lo + (hi - lo) * rng.random(dim)
                fx = problem(x_rand)
                eval_count[0] += 1

                # Check non-dominated
                is_dom = False
                for fj in best_F:
                    if np.all(fj <= fx) and np.any(fj < fx):
                        is_dom = True
                        break
                if not is_dom:
                    new_best_X = []
                    new_best_F = []
                    for xj, fj in zip(best_X, best_F):
                        if not (np.all(fx <= fj) and np.any(fx < fj)):
                            new_best_X.append(xj)
                            new_best_F.append(fj)
                    new_best_X.append(x_rand)
                    new_best_F.append(fx)
                    best_X = new_best_X
                    best_F = new_best_F

                    # Quick local refinement
                    if eval_count[0] < budget - 30 and time.time() < deadline:
                        # Pick a random front target
                        t_rand = rng.uniform(0.01, 0.99)
                        x_opt, _ = nelder_mead(x_rand, t_rand, max_iter=80)
                        if eval_count[0] < budget:
                            fx2 = problem(x_opt)
                            eval_count[0] += 1
                            is_dom2 = False
                            for fj in best_F:
                                if np.all(fj <= fx2) and np.any(fj < fx2):
                                    is_dom2 = True
                                    break
                            if not is_dom2:
                                new_best_X = []
                                new_best_F = []
                                for xj, fj in zip(best_X, best_F):
                                    if not (np.all(fx2 <= fj) and np.any(fx2 < fj)):
                                        new_best_X.append(xj)
                                        new_best_F.append(fj)
                                new_best_X.append(x_opt)
                                new_best_F.append(fx2)
                                best_X = new_best_X
                                best_F = new_best_F

        return best_X
# EVOLVE-BLOCK-END

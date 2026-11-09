import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Time guard: evaluator kills at 330s, leave safety margin
        start_time = time.time()
        deadline = start_time + 285

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Diverse initial sampling to find promising regions
        # The step function is flat until close to optimum, so we need
        # to find where the "cliff" is. Use strategic + random points.
        n_initial = min(30, budget)
        initial_points = []

        # Center
        initial_points.append(np.zeros(dim))

        # Axis-aligned points at various distances (explores the ellipsoidal structure)
        for i in range(dim):
            for dist in [1.0, 2.0, 3.0]:
                p = np.zeros(dim)
                p[i] = dist
                initial_points.append(p)
                p2 = np.zeros(dim)
                p2[i] = -dist
                initial_points.append(p2)

        # Random points to fill up to n_initial
        while len(initial_points) < n_initial:
            initial_points.append(rng.uniform(-4.5, 4.5, size=dim))

        initial_points = initial_points[:n_initial]

        # Evaluate initial points
        for p in initial_points:
            if evals_used >= budget or time.time() > deadline:
                break
            try_func(p)

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead simplex method
        # Direct, model-free geometric approach that exploits the step-ellipsoidal
        # structure by rapidly contracting simplices around the discontinuity.
        remaining = budget - evals_used
        if remaining < 20:
            return best_x, best_f

        def nelder_mead(x0, budget_alloc):
            nonlocal evals_used, best_f, best_x

            def nm_eval(x):
                if evals_used >= budget or time.time() > deadline:
                    return np.inf
                x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                f_val = func(x)
                if not np.isfinite(f_val):
                    return np.inf
                evals_used += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val

            # Initialize simplex
            alpha = 0.5
            simplex = [x0.copy()]
            f_vals = [nm_eval(x0)]
            evals = 1

            for i in range(dim):
                if evals >= budget_alloc or time.time() > deadline:
                    break
                xi = x0.copy()
                xi[i] += alpha if xi[i] < 4.5 else -alpha
                simplex.append(xi.copy())
                f_vals.append(nm_eval(xi))
                evals += 1

            if evals >= budget_alloc or time.time() > deadline:
                return

            # Nelder-Mead parameters
            reflection = 1.0
            expansion = 2.0
            contraction = 0.5
            shrink = 0.5

            # Track stagnation: if no improvement for many iterations, shrink more aggressively
            best_in_nm = min(f_vals)
            stagnation_count = 0

            while evals < budget_alloc and time.time() < deadline:
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i].copy() for i in order]
                f_vals = [f_vals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + reflection * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = nm_eval(x_ref)
                evals += 1
                if np.isinf(f_ref):
                    break

                if f_vals[0] <= f_ref < f_vals[-2]:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                elif f_ref < f_vals[0]:
                    # Expansion
                    x_exp = centroid + expansion * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = nm_eval(x_exp)
                    evals += 1
                    if np.isinf(f_exp):
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_vals[-1]:
                        x_con = centroid + contraction * (x_ref - centroid)
                    else:
                        x_con = centroid - contraction * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = nm_eval(x_con)
                    evals += 1
                    if np.isinf(f_con):
                        break

                    if f_con < min(f_ref, f_vals[-1]):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if evals >= budget_alloc or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_vals[i] = nm_eval(simplex[i])
                            evals += 1
                            if np.isinf(f_vals[i]):
                                break
                        if evals >= budget_alloc or time.time() > deadline:
                            break

                # Check for improvement
                current_best = min(f_vals)
                if current_best < best_in_nm - 1e-10:
                    best_in_nm = current_best
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # If stagnating, increase shrink rate to force convergence
                if stagnation_count > 5:
                    shrink = max(0.1, shrink * 0.7)
                    if shrink < 0.15:
                        break

        # Run multiple Nelder-Mead starts with weighted budget allocation
        # First start gets the most budget (best point), others get less
        n_starts = 5
        # Weighted allocation: first start gets 40%, others split remaining 60%
        weights = [0.4] + [0.12] * (n_starts - 1)
        # Normalize
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        for s in range(n_starts):
            if evals_used >= budget or time.time() > deadline:
                break

            budget_per_start = int(remaining * weights[s])
            if budget_per_start < 10:
                break

            if s == 0:
                x_start = best_x.copy()
            else:
                # Perturb the best point for diverse starts
                perturbation = rng.uniform(-1.0, 1.0, size=dim)
                x_start = np.clip(best_x + perturbation, -5.0, 5.0)

            nelder_mead(x_start, budget_per_start)

            # If no improvement from this start, reduce remaining budget allocation
            if evals_used >= budget or time.time() > deadline:
                break

        return best_x, best_f

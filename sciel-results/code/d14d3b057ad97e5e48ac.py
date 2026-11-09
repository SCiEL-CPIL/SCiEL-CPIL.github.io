"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: LHS spread + Nelder-Mead local refinement on non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Latin Hypercube Sampling for diverse initial spread ---
        n_lhs = min(budget, 60)  # use up to 60 points for LHS
        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_points[:, d] = (perm + np.random.random(n_lhs)) / n_lhs
        lhs_points = lo + span * lhs_points

        # Evaluate LHS points
        F = np.array([problem(x) for x in lhs_points])
        used = n_lhs

        # Track all evaluated points and their objectives
        all_X = list(lhs_points)
        all_F = list(F)

        # --- Identify non-dominated points for local refinement ---
        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for other in F_set:
                if np.all(other <= f) and np.any(other < f):
                    return True
            return False

        # Get non-dominated points from initial LHS
        non_dom_indices = []
        for i in range(n_lhs):
            if not is_dominated(F[i], list(F)):
                non_dom_indices.append(i)

        # If no non-dominated found (shouldn't happen), use all
        if not non_dom_indices:
            non_dom_indices = list(range(n_lhs))

        # Select up to 10 non-dominated points for Nelder-Mead refinement
        max_refine = min(10, len(non_dom_indices))
        # Sort by sum of objectives and pick the best ones
        non_dom_indices.sort(key=lambda i: np.sum(F[i]))
        refine_indices = non_dom_indices[:max_refine]

        # --- Nelder-Mead local refinement on each selected point ---
        def nelder_mead(x0, f0, max_iter=50):
            """Simple Nelder-Mead minimization for multi-objective (minimize sum)."""
            n = dim
            # Initial simplex
            simplex = [x0.copy()]
            f_vals = [f0.copy()]
            for i in range(n):
                perturbation = x0.copy()
                step = 0.1 * span[i] if span[i] != 0 else 0.1
                perturbation[i] += step
                perturbation = np.clip(perturbation, lo, hi)
                simplex.append(perturbation)
                f_vals.append(problem(perturbation))
                used_evals[0] += 1
                if used_evals[0] >= budget:
                    return x0, f0

            # Nelder-Mead iterations
            for _ in range(max_iter):
                if used_evals[0] >= budget or time.time() > deadline:
                    break
                # Sort by sum of objectives
                order = np.argsort([np.sum(f) for f in f_vals])
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = problem(xr)
                used_evals[0] += 1
                if used_evals[0] >= budget:
                    break

                # Check if reflection is good
                if np.sum(fr) < np.sum(f_vals[0]):
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe = problem(xe)
                    used_evals[0] += 1
                    if used_evals[0] >= budget:
                        break
                    if np.sum(fe) < np.sum(fr):
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif np.sum(fr) < np.sum(f_vals[-2]):
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = problem(xc)
                    used_evals[0] += 1
                    if used_evals[0] >= budget:
                        break
                    if np.sum(fc) < np.sum(f_vals[-1]):
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = (simplex[0] + simplex[i]) / 2.0
                            f_vals[i] = problem(simplex[i])
                            used_evals[0] += 1
                            if used_evals[0] >= budget:
                                break
                        if used_evals[0] >= budget:
                            break

                # Return best point found
            best_idx = np.argmin([np.sum(f) for f in f_vals])
            return simplex[best_idx], f_vals[best_idx]

        used_evals = [used]  # mutable counter

        for idx in refine_indices:
            if used_evals[0] >= budget or time.time() > deadline:
                break
            x_best, f_best = nelder_mead(all_X[idx].copy(), all_F[idx].copy())
            all_X.append(x_best)
            all_F.append(f_best)

        # --- Additional random sampling if budget remains ---
        while used_evals[0] < budget and time.time() < deadline:
            x = lo + span * np.random.random(dim)
            f = problem(x)
            used_evals[0] += 1
            all_X.append(x)
            all_F.append(f)

        return all_X
# EVOLVE-BLOCK-END

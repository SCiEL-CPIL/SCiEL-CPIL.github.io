"""Multi-objective optimizer for WFG7: Sobol stratified init + Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead with Sobol quasi-random initialization for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 270.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Use Sobol quasi-random points for stratified initialization
        # scipy.stats.qmc.Sobol is available in SciPy >= 1.7
        from scipy.stats.qmc import Sobol
        sobol = Sobol(d=dim, scramble=True, seed=42)
        n_initial = 30  # Number of initial simplex starts
        # Generate quasi-random points in [0, 1]^dim
        sobol_points = sobol.random(n_initial)
        X_init = lo + span * sobol_points
        F_init = np.array([problem(x) for x in X_init], dtype=float)
        used = n_initial

        # Keep track of all evaluated points for final hypervolume
        all_X = [X_init.copy()]
        all_F = [F_init.copy()]

        def dominates(a, b):
            """True if a dominates b."""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(F, idx):
            """Check if point idx is non-dominated in F."""
            f = F[idx]
            for j in range(len(F)):
                if j == idx:
                    continue
                if dominates(F[j], f):
                    return False
            return True

        # Nelder-Mead local search for each initial point
        # We have budget - n_initial evaluations left for local search
        remaining = budget - n_initial
        # Distribute remaining budget among the initial points
        # Give more budget to points that are currently non-dominated
        # First, identify non-dominated initial points
        nd_indices = [i for i in range(n_initial) if is_non_dominated(F_init, i)]
        
        # Allocate budget: give more to non-dominated points
        if len(nd_indices) > 0:
            # 70% of remaining budget to non-dominated, 30% to others
            nd_budget = int(remaining * 0.7)
            dom_budget = remaining - nd_budget
            per_nd = max(1, nd_budget // max(1, len(nd_indices)))
            per_dom = max(1, dom_budget // max(1, n_initial - len(nd_indices))) if n_initial > len(nd_indices) else 0
        else:
            per_nd = 0
            per_dom = max(1, remaining // n_initial)

        # Sort initial points: non-dominated first, then by sum of objectives
        order = sorted(range(n_initial), key=lambda i: (not is_non_dominated(F_init, i), np.sum(F_init[i])))

        for start_idx in order:
            if used >= budget:
                break
            if time.perf_counter() > deadline:
                break

            # Determine how many evaluations to give this start
            if is_non_dominated(F_init, start_idx):
                alloc = min(per_nd, budget - used)
            else:
                alloc = min(per_dom, budget - used)
            
            if alloc < 2:
                continue

            # Set up Nelder-Mead simplex
            # Initial simplex: start point + dim perturbed points
            base = X_init[start_idx].copy()
            base_f = F_init[start_idx].copy()

            # Create simplex
            simplex_X = np.zeros((dim + 1, dim))
            simplex_F = np.zeros((dim + 1, n_obj))
            simplex_X[0] = base
            simplex_F[0] = base_f
            simplex_used = 1

            # Generate perturbed points
            step_size = 0.1 * span
            for k in range(dim):
                if used >= budget or time.perf_counter() > deadline:
                    break
                pt = base.copy()
                pt[k] += step_size[k]
                pt = np.clip(pt, lo, hi)
                simplex_X[k + 1] = pt
                simplex_F[k + 1] = problem(pt)
                used += 1
                simplex_used += 1
                all_X.append(pt.reshape(1, -1))
                all_F.append(simplex_F[k + 1].reshape(1, -1))

            if simplex_used < 2:
                continue

            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            max_iters = alloc - simplex_used
            for it in range(max_iters):
                if used >= budget or time.perf_counter() > deadline:
                    break

                # Sort simplex by sum of objectives (proxy for quality)
                sums = np.sum(simplex_F[:simplex_used], axis=1)
                order_sm = np.argsort(sums)
                simplex_X[:simplex_used] = simplex_X[order_sm]
                simplex_F[:simplex_used] = simplex_F[order_sm]

                # Centroid of all but worst
                centroid = np.mean(simplex_X[:simplex_used - 1], axis=0)
                worst = simplex_X[simplex_used - 1]
                worst_f = simplex_F[simplex_used - 1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, lo, hi)
                f_r = problem(x_r)
                used += 1
                all_X.append(x_r.reshape(1, -1))
                all_F.append(f_r.reshape(1, -1))

                # Check if reflection is better than best
                if np.sum(f_r) < np.sum(simplex_F[0]):
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = problem(x_e)
                    used += 1
                    all_X.append(x_e.reshape(1, -1))
                    all_F.append(f_e.reshape(1, -1))
                    
                    if np.sum(f_e) < np.sum(f_r):
                        simplex_X[simplex_used - 1] = x_e
                        simplex_F[simplex_used - 1] = f_e
                    else:
                        simplex_X[simplex_used - 1] = x_r
                        simplex_F[simplex_used - 1] = f_r
                elif np.sum(f_r) < np.sum(worst_f):
                    # Reflection is better than worst but not best
                    simplex_X[simplex_used - 1] = x_r
                    simplex_F[simplex_used - 1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = problem(x_c)
                    used += 1
                    all_X.append(x_c.reshape(1, -1))
                    all_F.append(f_c.reshape(1, -1))
                    
                    if np.sum(f_c) < np.sum(worst_f):
                        simplex_X[simplex_used - 1] = x_c
                        simplex_F[simplex_used - 1] = f_c
                    else:
                        # Shrink
                        for i in range(1, simplex_used):
                            simplex_X[i] = simplex_X[0] + sigma * (simplex_X[i] - simplex_X[0])
                            simplex_X[i] = np.clip(simplex_X[i], lo, hi)
                            simplex_F[i] = problem(simplex_X[i])
                            used += 1
                            all_X.append(simplex_X[i].reshape(1, -1))
                            all_F.append(simplex_F[i].reshape(1, -1))
                            if used >= budget or time.perf_counter() > deadline:
                                break

            # After local search, keep the best point from this simplex
            if simplex_used > 0:
                best_local = np.argmin(np.sum(simplex_F[:simplex_used], axis=1))
                best_x = simplex_X[best_local]
                best_f = simplex_F[best_local]
                # Add to global population if not already there
                # We'll collect all points at the end

        # Collect all evaluated points
        if all_X:
            all_X = np.vstack(all_X)
            all_F = np.vstack(all_F)
        else:
            all_X = X_init
            all_F = F_init

        # Remove dominated points to get the Pareto front approximation
        n_total = len(all_F)
        front_mask = np.ones(n_total, dtype=bool)
        for i in range(n_total):
            if not front_mask[i]:
                continue
            for j in range(n_total):
                if i == j or not front_mask[j]:
                    continue
                if dominates(all_F[j], all_F[i]):
                    front_mask[i] = False
                    break

        front_X = all_X[front_mask]
        front_F = all_F[front_mask]

        return front_X
# EVOLVE-BLOCK-END

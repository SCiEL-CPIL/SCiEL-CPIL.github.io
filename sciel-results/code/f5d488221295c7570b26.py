import numpy as np
import time


class Optimizer:
    """Global-to-local Nelder-Mead simplex optimizer with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        rng = np.random.default_rng(12345)
        deadline = time.time() + 285.0
        calls = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Global search with multiple random restarts of Nelder-Mead
        # Use about 60% of budget for global phase
        global_budget = int(self.budget * 0.6)
        restarts = 8
        per_restart = max(20, global_budget // restarts)

        # Initial simplex size
        initial_alpha = 1.0

        for r in range(restarts):
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break

            # Start from a random point in the box, or near best if we have one
            if r == 0:
                x0 = np.zeros(n)
            else:
                # Perturb around current best with decreasing perturbation
                scale = 2.0 / (r + 1)
                x0 = best_x + scale * rng.standard_normal(n)
                x0 = np.clip(x0, -5.0, 5.0)

            # Build initial simplex
            simplex = []
            f0 = eval_point(x0)
            simplex.append((x0.copy(), f0))

            for i in range(n):
                xi = x0.copy()
                if f0 != 0:
                    xi[i] += initial_alpha * f0
                else:
                    xi[i] += initial_alpha
                xi = np.clip(xi, -5.0, 5.0)
                fi = eval_point(xi)
                simplex.append((xi, fi))

            # Nelder-Mead parameters
            alpha_ref = 1.0
            gamma_exp = 2.0
            rho_con = 0.5

            # Run Nelder-Mead iterations
            nm_iter_budget = per_restart
            for it in range(nm_iter_budget):
                if calls >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by fitness
                simplex.sort(key=lambda t: t[1])

                # Check convergence
                if np.max([abs(t[1] - simplex[0][1]) for t in simplex]) < 1e-12:
                    break
                if np.max([np.linalg.norm(t[0] - simplex[0][0]) for t in simplex]) < 1e-10:
                    break

                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_best = simplex[0][0]
                f_best = simplex[0][1]

                # Centroid of all points except worst
                centroid = np.mean([t[0] for t in simplex[:-1]], axis=0)

                # Reflection
                x_ref = centroid + alpha_ref * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = eval_point(x_ref)

                if f_ref < f_best:
                    # Expansion
                    x_exp = centroid + gamma_exp * (centroid - x_worst)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = eval_point(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = (x_exp, f_exp)
                    else:
                        simplex[-1] = (x_ref, f_ref)
                elif f_ref < simplex[-2][1]:
                    simplex[-1] = (x_ref, f_ref)
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid + 0.5 * (x_worst - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = eval_point(x_con)

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = (x_con, f_con)
                    else:
                        # Shrink
                        new_simplex = []
                        for i in range(len(simplex)):
                            if i == 0:
                                new_simplex.append((x_best, f_best))
                            else:
                                xi = x_best + rho_con * (simplex[i][0] - x_best)
                                xi = np.clip(xi, -5.0, 5.0)
                                fi = eval_point(xi)
                                new_simplex.append((xi, fi))
                        simplex = new_simplex

        # Phase 2: Local polish with derivative-free line search
        polish_budget = self.budget - calls
        if polish_budget > 5 and best_x is not None:
            # Use a pattern search / coordinate descent hybrid for final polish
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
            for ss in step_sizes:
                if calls >= self.budget:
                    break
                if time.time() > deadline:
                    break
                improved = False
                for dim_idx in range(n):
                    if calls >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        if calls >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        x_try = best_x.copy()
                        x_try[dim_idx] += direction * ss
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = eval_point(x_try)
                        if f_try < best_f:
                            improved = True
                            # Keep going in this direction
                            best_x = x_try.copy()
                            best_f = f_try
                            # Try to continue
                            x_try2 = x_try.copy()
                            x_try2[dim_idx] += direction * ss
                            x_try2 = np.clip(x_try2, -5.0, 5.0)
                            f_try2 = eval_point(x_try2)
                            if f_try2 < best_f:
                                best_x = x_try2.copy()
                                best_f = f_try2

                if not improved:
                    # No improvement at this scale, move to smaller
                    break

        # Phase 3: Final Nelder-Mead polish from best point
        remaining = self.budget - calls
        if remaining > 2 * n + 5 and best_x is not None:
            if time.time() < deadline:
                # Small simplex around best
                small_alpha = 0.05
                simplex = []
                f0 = eval_point(best_x)
                simplex.append((best_x.copy(), f0))
                for i in range(n):
                    xi = best_x.copy()
                    xi[i] += small_alpha
                    xi = np.clip(xi, -5.0, 5.0)
                    fi = eval_point(xi)
                    simplex.append((xi, fi))

                # Quick Nelder-Mead
                alpha_ref = 1.0
                gamma_exp = 2.0
                rho_con = 0.5
                max_iters = min(remaining - n - 1, 100)
                for it in range(max_iters):
                    if calls >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    simplex.sort(key=lambda t: t[1])
                    if np.max([abs(t[1] - simplex[0][1]) for t in simplex]) < 1e-14:
                        break

                    x_worst = simplex[-1][0]
                    f_worst = simplex[-1][1]
                    x_best = simplex[0][0]
                    f_best = simplex[0][1]
                    centroid = np.mean([t[0] for t in simplex[:-1]], axis=0)

                    x_ref = centroid + alpha_ref * (centroid - x_worst)
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = eval_point(x_ref)

                    if f_ref < f_best:
                        x_exp = centroid + gamma_exp * (centroid - x_worst)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = eval_point(x_exp)
                        if f_exp < f_ref:
                            simplex[-1] = (x_exp, f_exp)
                        else:
                            simplex[-1] = (x_ref, f_ref)
                    elif f_ref < simplex[-2][1]:
                        simplex[-1] = (x_ref, f_ref)
                    else:
                        if f_ref < f_worst:
                            x_con = centroid + 0.5 * (x_ref - centroid)
                        else:
                            x_con = centroid + 0.5 * (x_worst - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = eval_point(x_con)
                        if f_con < min(f_ref, f_worst):
                            simplex[-1] = (x_con, f_con)
                        else:
                            new_simplex = []
                            for i in range(len(simplex)):
                                if i == 0:
                                    new_simplex.append((x_best, f_best))
                                else:
                                    xi = x_best + rho_con * (simplex[i][0] - x_best)
                                    xi = np.clip(xi, -5.0, 5.0)
                                    fi = eval_point(xi)
                                    new_simplex.append((xi, fi))
                            simplex = new_simplex

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f

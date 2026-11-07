"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """Multistart Nelder-Mead with random restarts and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget

        # State
        count = 0
        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f, count
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def eval_point(x):
            nonlocal count
            try:
                f = func(x)
                count += 1
                return f
            except Exception:
                return float('inf')

        # Initial point: center of the box
        x0 = np.zeros(dim)
        f0 = eval_point(x0)
        if count == 0:
            return x0, float('inf')
        update_best(x0, f0)

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrinkage

        # Initial simplex: small perturbations around x0
        # Use a reasonable initial step size
        init_step = 1.0

        while count < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate a new starting point for multistart
            # If we've made good progress, start near the best point
            # Otherwise, try a random point in the box
            if best_f < 10.0:
                # Close to optimum, start near best
                x_start = best_x + np.random.randn(dim) * 0.5
            else:
                # Try a random point, but biased towards center
                x_start = np.random.uniform(-4, 4, dim)

            # Clip to box
            x_start = np.clip(x_start, -5, 5)

            # Build initial simplex
            simplex = [x_start.copy()]
            f_simplex = [eval_point(x_start)]
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = x_start.copy()

            # Add dim more points
            for i in range(dim):
                x_new = x_start.copy()
                # Perturb in direction i
                if x_new[i] + init_step > 5.0:
                    x_new[i] -= init_step
                else:
                    x_new[i] += init_step
                x_new = np.clip(x_new, -5, 5)
                simplex.append(x_new)
                f_simplex.append(eval_point(x_new))
                if f_simplex[-1] < best_f:
                    best_f = f_simplex[-1]
                    best_x = x_new.copy()

            # Nelder-Mead iterations
            max_nm_iters = 50
            for _ in range(max_nm_iters):
                if count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by fitness
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Check convergence
                if len(set(f_simplex)) == 1:
                    break
                f_range = f_simplex[-1] - f_simplex[0]
                if f_range < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_point(x_ref)
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_point(x_exp)
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_point(x_con)
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    if f_con < f_simplex[-1]:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = eval_point(simplex[i])
                            if f_simplex[i] < best_f:
                                best_f = f_simplex[i]
                                best_x = simplex[i].copy()

        return best_x, best_f

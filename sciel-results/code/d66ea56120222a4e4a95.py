import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            fx = func(np.clip(x, -5, 5))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5, 5).copy()
            return fx

        # Phase 1: Adaptive initialization - start at center, then explore
        start_x = np.zeros(dim)
        fx = call_func(start_x)
        init_points = [(fx, start_x.copy())]
        
        # Try random points to find a better starting location
        n_init = 12
        for _ in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx2 = call_func(x)
            init_points.append((fx2, x.copy()))

        if len(init_points) == 0:
            return best_x, best_f

        init_points.sort(key=lambda t: t[0])
        start_x = init_points[0][1].copy()
        best_f = init_points[0][0]

        # Phase 2: Nelder-Mead simplex method
        # Initialize simplex with adaptive step size
        simplex = [start_x.copy()]
        simplex_f = [best_f]

        # Use a moderate step size; for dim=5, 0.8 works well
        step_size = 0.8
        for i in range(dim):
            v = start_x.copy()
            v[i] += step_size
            fv = call_func(v)
            simplex.append(v.copy())
            simplex_f.append(fv)

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink
        max_iter = 300
        tol = 1e-10
        no_improve_count = 0
        best_f_nm = best_f

        for iteration in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break

            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]

            # Check convergence
            if abs(simplex_f[-1] - simplex_f[0]) < tol:
                break

            # Track no-improvement for restart
            if simplex_f[0] < best_f_nm - 1e-15:
                no_improve_count = 0
                best_f_nm = simplex_f[0]
            else:
                no_improve_count += 1

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = call_func(x_ref)

            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                f_exp = call_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (centroid - simplex[-1])
                f_con = call_func(x_con)
                if f_con < min(f_ref, simplex_f[-1]):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex_f[i] = call_func(simplex[i])

            # Adaptive restart: if stuck, try a new random point closer to current best
            if no_improve_count >= 25 and calls_made < budget - 15:
                # Try points near current best with decreasing perturbation
                for attempt in range(3):
                    if calls_made >= budget - 15:
                        break
                    perturbation = max(0.3, 1.5 * (0.7 ** attempt))
                    x_new = start_x + rng.normal(0, perturbation, dim)
                    x_new = np.clip(x_new, -4.5, 4.5)
                    f_new = call_func(x_new)
                    if f_new < simplex_f[0]:
                        simplex = [x_new.copy()]
                        simplex_f = [f_new]
                        new_step = max(0.4, step_size * 0.7)
                        for i in range(dim):
                            v = x_new.copy()
                            v[i] += new_step
                            fv = call_func(v)
                            simplex.append(v.copy())
                            simplex_f.append(fv)
                        step_size = new_step
                        no_improve_count = 0
                        best_f_nm = f_new
                        break
                else:
                    # If all attempts failed, reset no_improve counter to allow more iterations
                    no_improve_count = 15

        # Phase 3: Final polish - use remaining budget for high-precision local search
        remaining = budget - calls_made
        if remaining > 15 and time.time() < deadline - 5:
            # Do more Nelder-Mead iterations with tighter tolerance
            polish_tol = 1e-12
            for _ in range(min(remaining // 2, 80)):
                if calls_made >= budget or time.time() > deadline:
                    break
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                if abs(simplex_f[-1] - simplex_f[0]) < polish_tol:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = call_func(x_ref)
                
                if f_ref < simplex_f[0]:
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = call_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    if f_ref < simplex_f[-1]:
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        x_con = centroid - 0.5 * (centroid - simplex[-1])
                    f_con = call_func(x_con)
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = call_func(simplex[i])

        return best_x, best_f

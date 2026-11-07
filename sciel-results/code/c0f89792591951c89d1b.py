import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter = 200

        def run_nelder_mead(center, initial_step, random_init=False, max_iter_local=None):
            nonlocal best_x, best_f, calls_used

            if max_iter_local is None:
                max_iter_local = max_iter

            # Initialize simplex
            simplex = []
            simplex_f = []

            # First point is center
            x0 = clip(center)
            f0 = func(x0)
            calls_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            simplex.append(x0)
            simplex_f.append(f0)

            if calls_used >= budget - 5:
                return

            # Other points
            if random_init:
                for i in range(dim):
                    if calls_used >= budget - 5 or not time_ok():
                        return
                    direction = rng.normal(0, 1, dim)
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    x_i = center + initial_step * direction
                    x_i = clip(x_i)
                    f_i = func(x_i)
                    calls_used += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    simplex.append(x_i)
                    simplex_f.append(f_i)
            else:
                for i in range(dim):
                    if calls_used >= budget - 5 or not time_ok():
                        return
                    x_i = center.copy()
                    x_i[i] += initial_step
                    x_i = clip(x_i)
                    f_i = func(x_i)
                    calls_used += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    simplex.append(x_i)
                    simplex_f.append(f_i)

            # Main NM loop
            for _ in range(max_iter_local):
                if not time_ok() or calls_used >= budget - 5:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if max(simplex_f) - min(simplex_f) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = clip(x_r)
                f_r = func(x_r)
                calls_used += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if calls_used >= budget - 5 or not time_ok():
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()

                    if calls_used >= budget - 5 or not time_ok():
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if calls_used >= budget - 5 or not time_ok():
                            break

                        if f_c < f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5 or not time_ok():
                                    return
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if calls_used >= budget - 5 or not time_ok():
                            break

                        if f_c < worst_f:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5 or not time_ok():
                                    return
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s

        def polish_phase(center, steps=50):
            """Local random search with decreasing step size."""
            nonlocal best_x, best_f, calls_used

            if best_x is None:
                return

            x = clip(center).copy()
            f_curr = func(x)
            calls_used += 1
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()

            step_sizes = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            current_step_idx = 0
            no_improve_count = 0

            for _ in range(steps):
                if not time_ok() or calls_used >= budget - 5:
                    break

                step = step_sizes[current_step_idx]
                direction = rng.normal(0, 1, dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm

                x_new = x + step * direction
                x_new = clip(x_new)
                f_new = func(x_new)
                calls_used += 1

                if f_new < f_curr:
                    x = x_new
                    f_curr = f_new
                    no_improve_count = 0
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x.copy()
                else:
                    no_improve_count += 1
                    if no_improve_count > 10:
                        current_step_idx = min(current_step_idx + 1, len(step_sizes) - 1)
                        no_improve_count = 0

        # Phase 1: Initial exploration with Nelder-Mead from center
        if not time_ok() or calls_used >= budget - 5:
            return self.best_x, self.best_f

        run_nelder_mead(np.zeros(dim), initial_step=1.0, random_init=False, max_iter_local=80)

        # Phase 2: Second NM run with random initialization and smaller step
        if not time_ok() or calls_used >= budget - 5:
            return self.best_x, self.best_f

        if best_x is not None:
            run_nelder_mead(best_x, initial_step=0.5, random_init=True, max_iter_local=60)
        else:
            run_nelder_mead(rng.uniform(-3, 3, dim), initial_step=1.0, random_init=True, max_iter_local=60)

        # Phase 3: Third NM run with even smaller step for refined exploration
        if not time_ok() or calls_used >= budget - 5:
            return self.best_x, self.best_f

        if best_x is not None:
            run_nelder_mead(best_x, initial_step=0.2, random_init=True, max_iter_local=40)

        # Phase 4: Polish with local random search
        if not time_ok() or calls_used >= budget - 5:
            return self.best_x, self.best_f

        if best_x is not None:
            remaining_steps = min(50, max(0, budget - 5 - calls_used))
            if remaining_steps > 0:
                polish_phase(best_x, steps=remaining_steps)

        # Final check: ensure we have a valid result
        if best_x is None:
            # Fallback: just evaluate center
            x0 = np.zeros(dim)
            f0 = func(x0)
            calls_used += 1
            best_x = x0.copy()
            best_f = f0

        return best_x, best_f

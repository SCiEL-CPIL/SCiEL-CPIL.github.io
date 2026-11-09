import numpy as np
import time

class Optimizer:
    """Valley-aware adaptive coordinate descent + Nelder-Mead refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin for 300s limit

        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            if time.time() - start_time > time_limit:
                return best_f

            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start adaptive coordinate descent
        # Rosenbrock has a narrow curved valley; coordinate descent with
        # adaptive steps can follow it well.
        n_initializations = 6
        best_cd_x = None
        best_cd_f = float('inf')

        for init_idx in range(n_initializations):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break

            # Diverse starting points
            if init_idx == 0:
                x = np.zeros(self.dim)
            elif init_idx == 1:
                x = np.ones(self.dim) * 0.5
            else:
                x = rng.uniform(-3.5, 3.5, self.dim)

            fx = safe_eval(x)
            if fx >= best_cd_f:
                continue

            # Adaptive coordinate descent with step sizes
            step_sizes = np.ones(self.dim) * 1.0
            min_step = 1e-5
            max_step = 5.0
            stagnation_count = 0
            max_cd_iters = 150

            for cd_iter in range(max_cd_iters):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break

                improved = False

                for d in range(self.dim):
                    if calls_made >= self.budget or time.time() - start_time > time_limit:
                        break

                    # Try positive and negative steps
                    found_improvement = False
                    for sign in [1.0, -1.0]:
                        x_trial = x.copy()
                        x_trial[d] += sign * step_sizes[d]
                        x_trial = np.clip(x_trial, -5, 5)

                        fx_trial = safe_eval(x_trial)

                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            step_sizes[d] = min(step_sizes[d] * 1.5, max_step)
                            improved = True
                            found_improvement = True
                            break

                    if not found_improvement:
                        step_sizes[d] = max(step_sizes[d] * 0.5, min_step)

                # Update overall best
                if fx < best_cd_f:
                    best_cd_f = fx
                    best_cd_x = x.copy()

                # Check for convergence
                if not improved:
                    stagnation_count += 1
                    if stagnation_count >= 3:
                        break
                else:
                    stagnation_count = 0

                # Check if all steps are very small
                if np.all(step_sizes < min_step * 10):
                    break

        # Phase 2: Nelder-Mead refinement from best coordinate descent result
        if best_cd_x is not None and calls_made < self.budget:
            x0 = best_cd_x.copy()
            f0 = best_cd_f

            # Initial simplex with smaller steps since we're already in a good region
            step_size = 0.3
            simplex = [x0.copy()]
            for i in range(self.dim):
                v = x0.copy()
                v[i] += step_size
                simplex.append(v)

            f_values = [f0]
            for i in range(1, self.dim + 1):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                fx_i = safe_eval(simplex[i])
                f_values.append(fx_i)

            while len(f_values) < self.dim + 1:
                f_values.append(float('inf'))

            # Nelder-Mead Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            while calls_made < self.budget and time.time() - start_time < time_limit:
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]

                f_min = f_values[0]
                f_max = f_values[-1]
                if abs(f_max - f_min) < 1e-10 * max(1.0, abs(f_min)):
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)

                if f_r < f_values[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_eval(x_e)

                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)

                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_eval(x_c)

                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = safe_eval(simplex[i])

            # Update best if Nelder-Mead found something better
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()

        # Phase 3: If we still have budget, do a few more NM restarts from random points
        remaining_budget = self.budget - calls_made
        if remaining_budget > 30 and best_x is not None:
            n_extra_starts = min(3, remaining_budget // 30)
            for extra_idx in range(n_extra_starts):
                if calls_made >= self.budget - 10 or time.time() - start_time > time_limit - 5:
                    break

                # Start near the current best but perturbed
                x0 = best_x + rng.normal(0, 0.5, self.dim)
                x0 = np.clip(x0, -5, 5)

                # Quick NM
                step_size = 0.2
                simplex = [x0.copy()]
                for i in range(self.dim):
                    v = x0.copy()
                    v[i] += step_size
                    simplex.append(v)

                f_values = [safe_eval(simplex[0])]
                for i in range(1, self.dim + 1):
                    if calls_made >= self.budget or time.time() - start_time > time_limit:
                        break
                    f_values.append(safe_eval(simplex[i]))

                while len(f_values) < self.dim + 1:
                    f_values.append(float('inf'))

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5

                # Limit NM iterations for these extra starts
                max_nm_iters = 50
                for _ in range(max_nm_iters):
                    if calls_made >= self.budget or time.time() - start_time > time_limit:
                        break

                    indices = np.argsort(f_values)
                    simplex = [simplex[i] for i in indices]
                    f_values = [f_values[i] for i in indices]

                    f_min = f_values[0]
                    f_max = f_values[-1]
                    if abs(f_max - f_min) < 1e-10 * max(1.0, abs(f_min)):
                        break

                    centroid = np.mean(simplex[:-1], axis=0)

                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = safe_eval(x_r)

                    if f_r < f_values[0]:
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = safe_eval(x_e)

                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_values[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_values[-1] = f_r
                    elif f_r < f_values[-2]:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                    else:
                        if f_r < f_values[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (x_r - centroid)

                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)

                        if f_c < f_r:
                            simplex[-1] = x_c
                            f_values[-1] = f_c
                        else:
                            for i in range(1, self.dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_values[i] = safe_eval(simplex[i])

                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()

        # Fallback
        if best_x is None:
            best_x = np.ones(self.dim)
            best_f = func(np.clip(best_x, -5, 5))

        return best_x, best_f

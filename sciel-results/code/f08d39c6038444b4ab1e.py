import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(777)
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = float(fx)
                best_x = x.copy()
            return float(fx)

        def time_ok():
            return time.time() < deadline

        # ---- Phase 1: Diverse early sampling for fast initial progress ----
        # Mix of broad, medium, and near-origin samples to catch the basin early.
        samples = []
        # Broad uniform
        for _ in range(12):
            samples.append(rng.uniform(-4.5, 4.5, d))
        # Medium
        for _ in range(8):
            samples.append(rng.uniform(-2.5, 2.5, d))
        # Near origin (f14 often has min near shifted origin)
        for _ in range(6):
            samples.append(rng.uniform(-1.0, 1.0, d))
        # A few structured points
        samples.append(np.zeros(d))
        samples.append(rng.uniform(-5, 5, d))
        samples.append(rng.uniform(-5, 5, d))
        samples.append(rng.normal(0, 1.5, d))
        samples.append(rng.normal(0, 0.8, d))

        n_random = min(len(samples), budget // 8)
        for x in samples[:n_random]:
            if eval_count >= budget - 40 or not time_ok():
                break
            evaluate(x)

        # ---- Phase 2: Nelder-Mead for basin convergence ----
        step_size = 0.6
        simplex = [best_x.copy()]
        for i in range(d):
            v = best_x.copy()
            v[i] += step_size
            simplex.append(v)

        f_values = [evaluate(v) for v in simplex]

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        max_nm_iter = 120
        for _iter in range(max_nm_iter):
            if eval_count >= budget - 60 or not time_ok():
                break

            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]

            if f_values[-1] - f_values[0] < 1e-10 * (1 + abs(f_values[0])):
                break

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = f_values[-1]

            x_r = np.clip(centroid + alpha * (centroid - worst), -5, 5)
            f_r = evaluate(x_r)

            if f_r < f_values[0]:
                x_e = np.clip(centroid + gamma * (centroid - worst), -5, 5)
                f_e = evaluate(x_e)
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
                if f_r < f_worst:
                    x_c = np.clip(centroid + rho * (x_r - centroid), -5, 5)
                    f_c = evaluate(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, d + 1):
                            simplex[i] = np.clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]), -5, 5)
                            f_values[i] = evaluate(simplex[i])
                else:
                    x_c = np.clip(centroid - rho * (centroid - worst), -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, d + 1):
                            simplex[i] = np.clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]), -5, 5)
                            f_values[i] = evaluate(simplex[i])

        # ---- Phase 3: Exploit separability with per-coordinate line search ----
        # f14 is separable: each coordinate can be optimized independently given others.
        # Use a shrinking step schedule and bidirectional search per coordinate.
        refine_x = best_x.copy()
        refine_f = best_f

        step_schedule = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                         0.0078125, 0.00390625, 0.001953125, 0.0009765625]

        for step in step_schedule:
            if eval_count >= budget - 5 or not time_ok():
                break
            improved = False
            for i in range(d):
                if eval_count >= budget - 5 or not time_ok():
                    break
                # Bidirectional: try moving in both directions, keep best
                for direction in (1.0, -1.0):
                    if eval_count >= budget - 5 or not time_ok():
                        break
                    test_x = refine_x.copy()
                    test_x[i] += direction * step
                    test_x = np.clip(test_x, -5, 5)
                    fx = evaluate(test_x)
                    if fx < refine_f:
                        refine_f = fx
                        refine_x = test_x.copy()
                        best_f = fx
                        best_x = refine_x.copy()
                        improved = True
            if not improved:
                # No improvement at this step size; try a finer step once more
                break

        # ---- Phase 4: Fine coordinate refinement with bisection-like probing ----
        # A few more passes with very small steps to squeeze out the last bits.
        for step in [0.0005, 0.0002, 0.0001, 0.00005]:
            if eval_count >= budget - 2 or not time_ok():
                break
            for i in range(d):
                if eval_count >= budget - 2 or not time_ok():
                    break
                for direction in (1.0, -1.0):
                    if eval_count >= budget - 2 or not time_ok():
                        break
                    test_x = refine_x.copy()
                    test_x[i] += direction * step
                    test_x = np.clip(test_x, -5, 5)
                    fx = evaluate(test_x)
                    if fx < refine_f:
                        refine_f = fx
                        refine_x = test_x.copy()
                        best_f = fx
                        best_x = refine_x.copy()

        return best_x, best_f

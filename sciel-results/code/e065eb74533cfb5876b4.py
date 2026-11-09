"""Multi-start Nelder-Mead + adaptive (1+1)-ES for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead with adaptive (1+1)-ES refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_up():
            return time.time() > deadline or calls_made >= budget

        # ---------- Phase 1: diverse initialization (Halton + jitter) ----------
        n_init = 50

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        init_results = []
        for i in range(1, n_init + 1):
            if time_up():
                break
            point = []
            for d in range(dim):
                point.append(halton(i, bases[d % len(bases)]))
            point = np.array(point) * 10.0 - 5.0
            point += rng.uniform(-0.02, 0.02, dim)
            point = np.clip(point, -5, 5)
            f = evaluate(point)
            init_results.append((point, f))

        if not init_results:
            return best_x, best_f

        init_results.sort(key=lambda x: x[1])

        # ---------- Phase 2: Nelder-Mead from top starts ----------
        K = min(5, len(init_results))
        remaining = budget - calls_made
        budget_per_start = max(15, remaining // (K + 1))

        def nelder_mead(x0, f0, max_calls):
            """Run Nelder-Mead from x0, return best (x, f)."""
            if calls_made >= budget or time.time() > deadline:
                return x0, f0

            delta = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(np.clip(xi, -5, 5))

            simplex_f = [f0]
            for i in range(dim):
                if time_up():
                    return x0, f0
                simplex_f.append(evaluate(simplex[i + 1]))

            if len(simplex_f) < dim + 1:
                return x0, f0

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            no_improve = 0
            local_best = min(simplex_f)
            local_best_x = simplex[np.argmin(simplex_f)].copy()

            for _ in range(max_calls):
                if time_up():
                    break

                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]

                # convergence check
                max_dist = 0.0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.linalg.norm(simplex[i] - simplex[j])
                        if d > max_dist:
                            max_dist = d
                if max_dist < 1e-7:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = evaluate(xr)

                if fr < simplex_f[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1], simplex_f[-1] = xe, fe
                    else:
                        simplex[-1], simplex_f[-1] = xr, fr
                    no_improve = 0
                elif fr < simplex_f[-2]:
                    simplex[-1], simplex_f[-1] = xr, fr
                    no_improve = 0
                else:
                    if fr < simplex_f[-1]:
                        xc = np.clip(centroid + rho * (centroid - simplex[-1]), -5, 5)
                    else:
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    fc = evaluate(xc)
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1], simplex_f[-1] = xc, fc
                        no_improve = 0
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                            simplex_f[i] = evaluate(simplex[i])
                        no_improve += 1
                        if no_improve > 8:
                            break

                if simplex_f[0] < local_best:
                    local_best = simplex_f[0]
                    local_best_x = simplex[0].copy()

            return local_best_x, local_best

        for x0, f0 in init_results[:K]:
            if time_up():
                break
            nelder_mead(x0, f0, budget_per_start)

        # ---------- Phase 3: adaptive (1+1)-ES polish from global best ----------
        if best_x is not None and not time_up():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            success_window = []
            window_size = 15
            stagnation = 0
            max_stagnation = 60

            while not time_up():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    sigma *= 1.15
                    stagnation = 0
                else:
                    success_window.append(0)
                    sigma *= 0.92
                    stagnation += 1

                if len(success_window) >= window_size:
                    rate = np.mean(success_window)
                    if rate > 0.4:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.7
                    success_window = success_window[-window_size // 2:]

                sigma = np.clip(sigma, 0.005, 8.0)

                # escape shallow local minima
                if stagnation >= max_stagnation:
                    if rng.random() < 0.5:
                        x_new = np.clip(x + rng.standard_normal(dim) * 1.5, -5, 5)
                    else:
                        x_new = rng.uniform(-4, 4, dim)
                    fx_new = evaluate(x_new)
                    if fx_new < best_f:
                        x, fx = x_new.copy(), fx_new
                    else:
                        x, fx = best_x.copy(), best_f
                    stagnation = 0
                    sigma = 0.8
                    success_window = []

        return best_x, best_f
# EVOLVE-BLOCK-END

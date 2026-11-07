import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return func(x)

        def check_time():
            return time.time() < deadline - 2

        # Phase 1: Random sampling to find promising regions
        n_random = min(60, budget // 8)
        candidates = []
        for _ in range(n_random):
            if not check_time() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            candidates.append((f, x.copy()))

        # Sort candidates, keep top K
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))

        # Phase 2: Local refinement of top candidates using Nelder-Mead
        def nelder_mead(x0, max_iters=80, initial_step=0.5):
            nonlocal calls_made, best_f, best_x
            if not check_time() or calls_made >= budget:
                return x0, np.inf

            d = dim
            # Create simplex
            simplex = [x0.copy()]
            fs = [safe_func(x0)]
            for i in range(d):
                xi = x0.copy()
                xi[i] += initial_step
                simplex.append(xi)
                fs.append(safe_func(xi))
                if not check_time() or calls_made >= budget:
                    return simplex[0], fs[0]

            for it in range(max_iters):
                if not check_time() or calls_made >= budget:
                    break
                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fr = safe_func(xr)
                if not check_time() or calls_made >= budget:
                    break

                if fr < fs[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = safe_func(xe)
                    if not check_time() or calls_made >= budget:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = safe_func(xc)
                    if not check_time() or calls_made >= budget:
                        break
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fs[i] = safe_func(simplex[i])
                            if not check_time() or calls_made >= budget:
                                break

                # Update best
                min_idx = np.argmin(fs)
                if fs[min_idx] < best_f:
                    best_f = fs[min_idx]
                    best_x = simplex[min_idx].copy()

            # Return best from simplex
            min_idx = np.argmin(fs)
            return simplex[min_idx], fs[min_idx]

        # Refine top candidates
        for i in range(top_k):
            if not check_time() or calls_made >= budget:
                break
            f_init, x_init = candidates[i]
            x_opt, f_opt = nelder_mead(x_init, max_iters=60, initial_step=0.5)

            # Phase 3: Try to escape local minima by jumping
            # Rastrigin has periodicity ~2π, so try perturbations of size ~1
            if not check_time() or calls_made >= budget:
                continue

            # Try small random perturbations to find better basins
            for _ in range(10):
                if not check_time() or calls_made >= budget:
                    break
                x_trial = x_opt + rng.standard_normal(dim) * 0.8
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = safe_func(x_trial)
                if f_trial < f_opt:
                    x_opt = x_trial
                    f_opt = f_trial
                    # Quick local refinement
                    x_opt, f_opt = nelder_mead(x_opt, max_iters=20, initial_step=0.3)

        # Phase 4: Final polish with very fine search around best
        if best_x is not None and check_time() and calls_made < budget:
            # Fine coordinate descent
            step = 0.05
            x = best_x.copy()
            fx = best_f
            for _ in range(30):
                if not check_time() or calls_made >= budget:
                    break
                improved = False
                for d in range(dim):
                    if not check_time() or calls_made >= budget:
                        break
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        if np.allclose(x_trial, x):
                            continue
                        f_trial = safe_func(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.001:
                        break

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f

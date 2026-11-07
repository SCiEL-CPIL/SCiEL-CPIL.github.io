"""Improved optimizer for BBOB f22 using direct structural approach:
Halton-based broad sampling + Powell's method + Nelder-Mead local refinement."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin of 50s before 330s kill
        n = self.dim
        budget = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 2:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Halton sequence for stratified sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_vector(index, dims):
            point = []
            bases = [2, 3, 5, 7, 11]
            for d in range(dims):
                point.append(halton(index, bases[d]))
            return np.array(point)

        # Powell's method - conjugate direction optimization
        def powell_refine(x0, max_iter=30):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget or x0 is None:
                return best_x.copy(), best_f

            x = x0.copy()
            # Initial directions: coordinate axes
            directions = [np.eye(n, dtype=np.float64)[i] for i in range(n)]
            f_x = try_eval(x)
            if f_x is None:
                return best_x.copy(), best_f

            for iteration in range(max_iter):
                if total_calls >= budget or time.time() > deadline - 5:
                    break

                x_old = x.copy()
                f_old = f_x
                f_min = np.inf
                min_dir_idx = 0

                for i in range(n):
                    if total_calls >= budget or time.time() > deadline - 5:
                        return best_x.copy(), best_f

                    # Line search along direction i
                    best_a = 0.0
                    best_f_line = f_x
                    # Try a few step sizes
                    for a in [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.5, -0.5]:
                        x_test = x + a * directions[i]
                        x_test = np.clip(x_test, -5.0, 5.0)
                        f_test = try_eval(x_test)
                        if f_test is None:
                            break
                        if f_test < best_f_line:
                            best_f_line = f_test
                            best_a = a
                    if f_test is None:
                        break

                    if best_a != 0.0:
                        x = x + best_a * directions[i]
                        f_x = best_f_line

                    if f_x < f_min:
                        f_min = f_x
                        min_dir_idx = i

                # Check convergence
                if abs(f_old - f_x) < 1e-10:
                    break

                # Replace the direction that gave the largest decrease
                # with the new conjugate direction
                d_new = x - x_old
                if np.linalg.norm(d_new) > 1e-12:
                    directions[min_dir_idx] = d_new / np.linalg.norm(d_new)
                else:
                    break

            return best_x.copy(), best_f

        # Nelder-Mead from the best point found
        def nelder_mead(x0, max_calls=None):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget or x0 is None:
                return best_x.copy(), best_f

            # Build initial simplex
            alpha = 0.5
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += alpha if x0[i] < 4.0 else -alpha
                simplex.append(xi)

            # Evaluate initial simplex
            simplex_f = []
            for i in range(n + 1):
                if total_calls >= budget or time.time() > deadline - 5:
                    return best_x.copy(), best_f
                f_i = try_eval(simplex[i])
                if f_i is None:
                    return best_x.copy(), best_f
                simplex_f.append(f_i)

            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Nelder-Mead parameters
            gamma = 1.5  # reflection
            rho = 2.0    # expansion
            sigma_nm = 0.5  # contraction

            no_improve = 0
            patience = 20

            while total_calls < budget and time.time() < deadline - 5:
                # Check convergence
                f_range = simplex_f[-1] - simplex_f[0]
                x_range = max(np.linalg.norm(simplex[-1] - simplex[0]), 1e-12)
                if f_range < 1e-8 and x_range < 1e-6:
                    break
                if no_improve >= patience:
                    break

                # Centroid of all but worst point
                centroid = np.mean([simplex[i] for i in range(n)], axis=0)

                # Reflection
                x_r = centroid + gamma * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                f_r = try_eval(x_r)
                if f_r is None:
                    break

                if simplex_f[0] <= f_r < simplex_f[-1]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                    no_improve = 0 if f_r < simplex_f[-2] else no_improve + 1
                elif f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    f_e = try_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    no_improve = 0
                else:
                    # Contraction
                    x_c = centroid + sigma_nm * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                        no_improve += 1
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if total_calls >= budget or time.time() > deadline - 3:
                                return best_x.copy(), best_f
                            f_i = try_eval(simplex[i])
                            if f_i is None:
                                return best_x.copy(), best_f
                            simplex_f[i] = f_i
                        no_improve += 1

                # Re-sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

            # Return best point
            best_idx = int(np.argmin(simplex_f))
            if simplex_f[best_idx] < best_f:
                best_f = simplex_f[best_idx]
                best_x = simplex[best_idx].copy()
            return best_x.copy(), best_f

        # Phase 1: Stratified sampling with Halton sequence
        n_phase1 = min(60, budget // 5)
        phase1_results = []
        for i in range(1, n_phase1 + 1):
            if total_calls >= budget or time.time() > deadline - 10:
                break
            h_point = halton_vector(i, n)
            x_try = -4.5 + 9.0 * h_point
            f_try = try_eval(x_try)
            if f_try is None:
                break
            phase1_results.append((f_try, x_try.copy()))

        # Phase 1b: Add structured points
        structured = [
            np.zeros(n),
            np.ones(n) * 2.0,
            -np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.ones(n) * 3.0,
            -np.ones(n) * 3.0,
            np.ones(n) * 1.0,
            -np.ones(n) * 1.0,
            np.ones(n) * 0.5,
            -np.ones(n) * 0.5,
        ]
        for p in structured:
            if total_calls >= budget or time.time() > deadline - 10:
                break
            f_s = try_eval(p)
            if f_s is not None:
                phase1_results.append((f_s, p.copy()))

        # Sort phase1_results by function value
        if phase1_results:
            phase1_results.sort(key=lambda x: x[0])

        # Phase 2: Run Powell's method from top distinct points
        powell_starts = []
        for f_val, x_val in phase1_results[:5]:
            is_distinct = True
            for _, x_existing in powell_starts:
                if np.linalg.norm(x_val - x_existing) < 1.0:
                    is_distinct = False
                    break
            if is_distinct and total_calls < budget - 30:
                powell_starts.append((f_val, x_val))
            if len(powell_starts) >= 3:
                break

        for f_start, x_start in powell_starts:
            if total_calls < budget - 20 and time.time() < deadline - 15:
                powell_refine(x_start.copy())

        # Phase 3: Run Nelder-Mead from the best point after Powell
        if total_calls < budget - 10 and time.time() < deadline - 10:
            nelder_mead(best_x.copy())

        # Phase 4: Multi-scale coordinate descent polish
        if total_calls < budget and best_x is not None and time.time() < deadline - 5:
            for step in [0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]:
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    for direction in [-1, 1]:
                        x_try = best_x.copy()
                        x_try[coord] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_eval(x_try)
                        if f_try is None:
                            break
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            # Keep stepping in same direction while improving
                            for _ in range(5):
                                if total_calls >= budget or time.time() > deadline - 3:
                                    break
                                x_next = best_x.copy()
                                x_next[coord] += direction * step
                                x_next = np.clip(x_next, -5.0, 5.0)
                                f_next = try_eval(x_next)
                                if f_next is None:
                                    break
                                if f_next < best_f:
                                    best_f = f_next
                                    best_x = x_next.copy()
                                else:
                                    break

        # Phase 5: Final fine-grained local search around best point
        if total_calls < budget and best_x is not None and time.time() < deadline - 2:
            rng_final = np.random.default_rng(999)
            for scale in [0.03, 0.01, 0.005]:
                if total_calls >= budget or time.time() > deadline - 2:
                    break
                for _ in range(8):
                    if total_calls >= budget or time.time() > deadline - 2:
                        break
                    x_perturb = best_x + rng_final.normal(0, scale, n)
                    x_perturb = np.clip(x_perturb, -5.0, 5.0)
                    f_perturb = try_eval(x_perturb)
                    if f_perturb is not None and f_perturb < best_f:
                        best_f = f_perturb
                        best_x = x_perturb.copy()

        return best_x, best_f

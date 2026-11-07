# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0

        def _ok():
            return time.time() < deadline and evaluated < budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def _eval(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def _nelder_mead(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(42 + seed_off)
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            # Build initial simplex
            simplex = []
            simplex_f = []

            x0 = np.clip(np.asarray(x0, dtype=float), -5.0, 5.0)
            f0 = _eval(x0)
            simplex.append(x0.copy())
            simplex_f.append(f0)

            if not _ok():
                return

            # Add perturbation points
            for i in range(dim):
                if not _ok():
                    break
                xi = x0.copy()
                xi[i] += 0.5
                xi = np.clip(xi, -5.0, 5.0)
                fi = _eval(xi)
                simplex.append(xi)
                simplex_f.append(fi)

            if len(simplex) < 2:
                return

            # Main Nelder-Mead loop
            max_iter = alloc
            for _ in range(max_iter):
                if not _ok():
                    break

                # Sort by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Convergence check
                if max(simplex_f) - min(simplex_f) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = np.clip(centroid + alpha * (centroid - worst), -5.0, 5.0)
                f_r = _eval(x_r)

                if not _ok():
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = np.clip(centroid + gamma * (x_r - centroid), -5.0, 5.0)
                    f_e = _eval(x_e)
                    if not _ok():
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        x_c = np.clip(centroid + rho * (x_r - centroid), -5.0, 5.0)
                        f_c = _eval(x_c)
                        if not _ok():
                            break
                        if f_c < f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_f[i] = _eval(simplex[i])
                    else:
                        x_c = np.clip(centroid - rho * (worst - centroid), -5.0, 5.0)
                        f_c = _eval(x_c)
                        if not _ok():
                            break
                        if f_c < worst_f:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_f[i] = _eval(simplex[i])

        def _coordinate_descent(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(77 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f

            for _ in range(alloc):
                if not _ok():
                    break
                improved = False
                for i in range(dim):
                    if not _ok():
                        break
                    step = 0.5
                    for _s in range(8):
                        if not _ok():
                            break
                        found = False
                        for sign in (1, -1):
                            x_try = x.copy()
                            x_try[i] += sign * step
                            x_try = np.clip(x_try, -5.0, 5.0)
                            f_try = _eval(x_try)
                            if f_try < fx:
                                fx = f_try
                                x = x_try
                                step *= 1.5
                                improved = True
                                found = True
                                break
                        if not found:
                            break
                    if not _ok():
                        break
                if not improved:
                    break

            # Track best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def _plus_one_es(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(99 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f
            sigma = 0.3

            while _ok():
                if evaluated >= budget:
                    break
                z = r.standard_normal(dim)
                x_try = np.clip(x + sigma * z, -5.0, 5.0)
                f_try = _eval(x_try)
                if f_try < fx:
                    fx = f_try
                    x = x_try
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    sigma = max(sigma * 0.8, 1e-6)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # === Strategy ===
        # Phase 1: Broad exploration with multiple Nelder-Mead starts
        n_starts = 4
        alloc1 = max(10, budget // 8)

        for i in range(n_starts):
            if not _ok():
                break
            start = rng.uniform(-4.0, 4.0, size=dim)
            _nelder_mead(start, alloc1, seed_off=i * 100)

        # Phase 2: Coordinate descent refinement from best
        if best_x is not None and _ok():
            alloc2 = max(10, budget // 6)
            _coordinate_descent(best_x, alloc2, seed_off=0)

        # Phase 3: Nelder-Mead refinement from best
        if best_x is not None and _ok():
            alloc3 = max(10, budget // 5)
            _nelder_mead(best_x, alloc3, seed_off=500)

        # Phase 4: (1+1)-ES polish
        if best_x is not None and _ok():
            remaining = budget - evaluated
            _plus_one_es(best_x, remaining, seed_off=200)

        return best_x, best_f
# EVOLVE-BLOCK-END

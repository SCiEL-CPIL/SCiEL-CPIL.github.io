"""Multi-start Nelder-Mead optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 280s margin within 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # --- Nelder-Mead simplex method (vectorized where possible) ---
        def nelder_mead(x0, f0, max_iter=200):
            nonlocal total_calls, best_f, best_x
            n = dim
            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(n):
                xi = x0.copy()
                if f0 > 0.1:
                    xi[i] += 0.5
                else:
                    xi[i] += 0.1
                simplex.append(np.clip(xi, -5, 5))
                fvals.append(safe_call(simplex[-1]))

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            for it in range(max_iter):
                if total_calls >= budget or time.time() > deadline:
                    break
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                    fc = safe_call(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                            fvals[i] = safe_call(simplex[i])
                        if total_calls >= budget or time.time() > deadline:
                            break

            return simplex[0], fvals[0]

        # --- Multi-start strategy ---
        # Phase 1: Coarse random sampling to identify promising regions
        n_coarse = 30
        coarse_points = []
        coarse_fvals = []
        for i in range(n_coarse):
            if total_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            coarse_points.append(x)
            coarse_fvals.append(fx)

        # Sort by function value, take top seeds
        order = np.argsort(coarse_fvals)
        n_starts = min(8, len(order))
        # Allocate remaining budget among starts
        remaining = budget - total_calls
        per_start = max(20, remaining // n_starts)

        # Phase 2: Nelder-Mead from each promising start
        for idx in order[:n_starts]:
            if total_calls >= budget or time.time() > deadline:
                break
            x0 = coarse_points[idx]
            f0 = coarse_fvals[idx]
            # Local NM
            nelder_mead(x0, f0, max_iter=per_start)

        # Phase 3: If budget remains, do a few more random restarts
        n_extra = 5
        for i in range(n_extra):
            if total_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            nelder_mead(x, fx, max_iter=min(50, budget - total_calls))

        # Final polish: one more NM from best point if budget allows
        if best_x is not None and total_calls < budget and time.time() < deadline:
            nelder_mead(best_x.copy(), best_f, max_iter=min(40, budget - total_calls))

        return best_x, best_f
# EVOLVE-BLOCK-END

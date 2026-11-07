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
        t0 = time.time()
        deadline = t0 + 280  # 20s safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, f0, remaining_budget, max_iters=200):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return
            n = dim
            # Build initial simplex
            simplex = [x0.copy()]
            vals = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 if xi[i] < 4.5 else -0.5
                fi = safe_call(xi)
                if calls >= budget:
                    break
                simplex.append(xi)
                vals.append(fi)
            if len(simplex) < 2:
                return

            alpha, gamma, rho, sigma = 1.0, 2.0, -0.5, 0.5
            iters = 0
            while iters < max_iters and calls < budget:
                if time.time() > deadline:
                    return
                iters += 1
                # Sort simplex by function value
                order = np.argsort(vals)
                simplex = [simplex[i] for i in order]
                vals = [vals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_call(xr)
                if calls >= budget:
                    return

                if fr < vals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if calls >= budget:
                        return
                    if fe < fr:
                        simplex[-1] = xe
                        vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        vals[-1] = fr
                elif fr < vals[-2]:
                    simplex[-1] = xr
                    vals[-1] = fr
                else:
                    # Contraction
                    if fr < vals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    fc = safe_call(xc)
                    if calls >= budget:
                        return
                    if fc < min(fr, vals[-1]):
                        simplex[-1] = xc
                        vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            vals[i] = safe_call(simplex[i])
                            if calls >= budget:
                                return

                # Check convergence
                if len(vals) >= 2:
                    spread = np.std([v for v in vals if np.isfinite(v)])
                    if spread < 1e-10:
                        break

        # Coarse screening: evaluate several starting points
        n_starts = 8
        starts = []
        # Include origin and some structured points
        starts.append(np.zeros(dim))
        starts.append(rng.uniform(-4, 4, dim))
        for _ in range(n_starts - 2):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        start_vals = []
        for s in starts:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            fs = safe_call(s)
            start_vals.append((fs, s))

        # Sort by function value and run Nelder-Mead from best starts
        start_vals.sort(key=lambda x: x[0])
        for fs, s in start_vals:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            remaining = budget - calls
            if remaining < 10:
                break
            nelder_mead(s, fs, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END

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
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Wrap func to count calls and guard against exceptions
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def safe_call(x):
            if call_count[0] >= budget:
                return best_f[0]
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                call_count[0] += 1
                if not np.isfinite(f):
                    f = 1e10
                if f < best_f[0]:
                    best_f[0] = f
                    best_x[0] = x.copy()
                return f
            except Exception:
                return 1e10

        def nelder_mead(x0, f0, max_iters):
            """Run Nelder-Mead from x0 with initial function value f0."""
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += 0.25

            fvals = np.zeros(n + 1)
            fvals[0] = f0
            for i in range(n):
                fvals[i + 1] = safe_call(simplex[i + 1])

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                if call_count[0] >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-12:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:n], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[n])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[n])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                elif fr < fvals[n - 1]:
                    simplex[n] = xr
                    fvals[n] = fr
                else:
                    # Contraction
                    if fr < fvals[n]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_call(xc)
                        if fc <= fr:
                            simplex[n] = xc
                            fvals[n] = fc
                        else:
                            break
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[n])
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_call(xc)
                        if fc <= fvals[n]:
                            simplex[n] = xc
                            fvals[n] = fc
                        else:
                            break

                # Shrink
                if np.max(fvals) - np.min(fvals) > 1e15:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = safe_call(simplex[i])

            return simplex[0], fvals[0]

        # Phase 1: Coarse screening with random points
        rng = np.random.default_rng(42)
        n_screen = 40
        screen_points = []
        for _ in range(n_screen):
            x = rng.uniform(-4.5, 4.5, dim)
            f = safe_call(x)
            screen_points.append((f, x))
            if call_count[0] >= budget or time.time() > deadline:
                break

        screen_points.sort(key=lambda t: t[0])

        # Phase 2: Local refinement with Nelder-Mead on best candidates
        n_local = min(8, len(screen_points))
        for i in range(n_local):
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break
            f0, x0 = screen_points[i]
            # Use remaining budget for this local search
            remaining = budget - call_count[0]
            max_iters = max(10, remaining // 5)
            _, _ = nelder_mead(x0, f0, max_iters)

        # Phase 3: If we still have budget, do a few more random starts + local
        if call_count[0] < budget and time.time() < deadline:
            n_more = min(5, (budget - call_count[0]) // 50)
            for _ in range(n_more):
                if time.time() > deadline or call_count[0] >= budget:
                    break
                x = rng.uniform(-4.5, 4.5, dim)
                f = safe_call(x)
                if f < best_f[0]:
                    best_x[0] = x.copy()
                    best_f[0] = f
                    remaining = budget - call_count[0]
                    if remaining > 20:
                        max_iters = max(10, remaining // 5)
                        nelder_mead(x, f, max_iters)

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END

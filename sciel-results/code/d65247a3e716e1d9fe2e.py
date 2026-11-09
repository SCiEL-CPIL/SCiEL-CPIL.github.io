"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, NelderMead, L_BFGS_B


class Optimizer:
    """Hybrid global-to-local optimizer: Halton global search + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Halton sequence generator for low-discrepancy global coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            """Generate n Halton points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for d in range(dim):
                    # Use first dim primes
                    base = 2 + _primes[d]
                    points[i - 1, d] = halton(i, base)
            return points

        # Precompute primes
        _primes = [0, 1, 3, 5, 7, 11, 13, 17, 19, 23]

        # Phase 1: Global search with Halton + random perturbations
        # Phase 2: Local polish on best points found

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            call_count += 1
            return func(x)

        # Phase 1: Halton-based global search
        # Use a mix of Halton and random points for robustness
        n_halton = min(200, budget // 3)
        n_random = min(100, budget // 4)

        # Generate Halton points mapped to [-5, 5]^dim
        halton_pts = halton_sequence(n_halton, dim)
        halton_pts = halton_pts * 10.0 - 5.0  # map to [-5, 5]

        # Also add some random points
        random_pts = rng.uniform(-5, 5, (n_random, dim))

        # Combine and evaluate
        all_global_pts = np.vstack([halton_pts, random_pts])

        # Shuffle to avoid bias in ordering
        order = rng.permutation(len(all_global_pts))
        all_global_pts = all_global_pts[order]

        for i, pt in enumerate(all_global_pts):
            if call_count >= budget:
                break
            if time.time() > deadline - 60:  # leave time for polish
                break

            f = safe_call(pt)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: Local polish using Nelder-Mead on best point
        if best_x is not None and call_count < budget and time.time() < deadline - 30:
            # Try Nelder-Mead from the best point
            try:
                # Count remaining budget
                remaining = budget - call_count
                if remaining > 20:
                    # Use a custom objective that tracks calls
                    nm_calls = [0]

                    def nm_obj(x):
                        nonlocal call_count, best_f, best_x
                        if call_count >= budget or time.time() > deadline - 10:
                            return best_f
                        call_count += 1
                        nm_calls[0] += 1
                        fx = func(x)
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        return fx

                    # Nelder-Mead parameters
                    initial_simplex = [best_x]
                    for d in range(dim):
                        point = best_x.copy()
                        point[d] += 0.5 if point[d] < 4.5 else -0.5
                        initial_simplex.append(np.clip(point, -5, 5))

                    res = minimize(
                        nm_obj,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 500),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )

                    if res.x is not None:
                        fx_final = func(np.clip(res.x, -5, 5)) if call_count < budget else best_f
                        if fx_final < best_f:
                            best_f = fx_final
                            best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: If budget remains, do a few more random restarts with quick polish
        if best_x is not None and call_count < budget and time.time() < deadline - 20:
            remaining = budget - call_count
            if remaining > 30:
                n_restarts = min(5, remaining // 15)
                for _ in range(n_restarts):
                    if time.time() > deadline - 10:
                        break
                    # Start near best with some perturbation
                    x0 = best_x + rng.normal(0, 1.0, dim)
                    x0 = np.clip(x0, -5, 5)
                    f0 = safe_call(x0)
                    if f0 is None:
                        break
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()

                    # Quick Nelder-Mead polish
                    if call_count < budget - 10:
                        try:
                            def nm_obj2(x):
                                nonlocal call_count, best_f, best_x
                                if call_count >= budget or time.time() > deadline - 5:
                                    return best_f
                                call_count += 1
                                fx = func(x)
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                return fx

                            initial_simplex2 = [x0]
                            for d in range(dim):
                                point = x0.copy()
                                point[d] += 0.3 if point[d] < 4.7 else -0.3
                                initial_simplex2.append(np.clip(point, -5, 5))

                            res2 = minimize(
                                nm_obj2,
                                x0,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': min(budget - call_count, 100),
                                    'xatol': 1e-8,
                                    'fatol': 1e-10,
                                    'adaptive': True
                                }
                            )
                        except Exception:
                            pass

        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END

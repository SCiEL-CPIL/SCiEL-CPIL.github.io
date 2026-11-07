"""Nelder-Mead simplex optimizer for BBOB Weierstrass f16.

Uses sequential Nelder-Mead simplex runs with adaptive restarts,
exploiting the local smoothness of the Weierstrass function for
rapid early convergence.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Nelder-Mead simplex with sequential restarts and polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(777)
        deadline = time.time() + 270.0
        n_calls = [0]

        def safe(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float, copy=True)

        # ---- Nelder-Mead simplex implementation ----
        def nelder_mead(start_x, max_calls=60):
            """Run Nelder-Mead from start_x, using up to max_calls evaluations."""
            nonlocal best_x, best_f, n_calls

            # Initialize simplex: dim+1 points
            simplex = [np.array(start_x, dtype=float)]
            fvals = [safe(start_x)]

            # Add dim perturbation points
            for i in range(dim):
                pt = np.array(start_x, dtype=float)
                step = 0.5 if abs(start_x[i]) < 1.0 else 0.1
                pt[i] += step
                pt = np.clip(pt, -5.0, 5.0)
                simplex.append(pt)
                fvals.append(safe(pt))

            # Track best
            for x, f in zip(simplex, fvals):
                if np.isfinite(f) and f < best_f:
                    record(x, f)

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = max_calls - (dim + 1)

            for it in range(max_iter):
                if n_calls[0] >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Check convergence: spread of function values
                if np.isfinite(fvals[0]) and np.isfinite(fvals[-1]):
                    spread = fvals[-1] - fvals[0]
                    if spread < 1e-10:
                        break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe(xr)

                if np.isfinite(fr) and fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe(xe)
                    if np.isfinite(fe) and fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif np.isfinite(fr) and fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if np.isfinite(fr) and fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + sigma * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - sigma * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe(xc)

                    if np.isfinite(fc) and fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe(simplex[i])

                # Update global best
                for x, f in zip(simplex, fvals):
                    if np.isfinite(f) and f < best_f:
                        record(x, f)

            return best_f

        # ---- Phase 1: Initial evaluation at origin ----
        f0 = safe(np.zeros(dim))
        if np.isfinite(f0):
            record(np.zeros(dim), f0)

        # ---- Phase 2: Sequential Nelder-Mead runs ----
        # First run at origin (exploits local smoothness)
        if n_calls[0] < budget and time.time() < deadline:
            nelder_mead(np.zeros(dim), max_calls=60)

        # Subsequent runs: alternate between random starts and perturbations of best
        run_count = 1
        while n_calls[0] < budget and time.time() < deadline and run_count < 15:
            run_count += 1

            # Choose start point
            if run_count % 3 == 0:
                # Random start in the box
                start_x = rng.uniform(-4.0, 4.0, dim)
            elif run_count % 3 == 1:
                # Perturbation of current best
                pert_scale = max(0.5, 2.0 / run_count)
                start_x = best_x + pert_scale * rng.standard_normal(dim)
            else:
                # Structured start
                base = rng.choice([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
                start_x = np.full(dim, base) + rng.uniform(-0.5, 0.5, dim)

            start_x = np.clip(start_x, -5.0, 5.0)

            # Evaluate start
            fs = safe(start_x)
            if np.isfinite(fs):
                record(start_x, fs)

            # Run Nelder-Mead from this start
            # Decrease max_calls as we go to leave budget for later runs
            remaining = budget - n_calls[0]
            if remaining < 10:
                break
            calls_for_run = min(50, max(15, remaining // 3))
            nelder_mead(start_x, max_calls=calls_for_run)

        # ---- Phase 3: Fine polish with small-step local search ----
        if n_calls[0] < budget and time.time() < deadline:
            polish_budget = budget - n_calls[0]
            sigma = 0.01
            for _ in range(polish_budget):
                if n_calls[0] >= budget or time.time() > deadline:
                    break
                cand = best_x + sigma * rng.standard_normal(dim)
                cand = np.clip(cand, -5.0, 5.0)
                f = safe(cand)
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = cand.copy()
                    sigma *= 0.98
                else:
                    sigma *= 0.99
                sigma = max(sigma, 1e-7)

        return best_x, best_f
# EVOLVE-BLOCK-END

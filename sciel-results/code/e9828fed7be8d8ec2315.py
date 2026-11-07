"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + Nelder-Mead refinement with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        # Track all evaluations
        call_count = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return float('inf')
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x_clipped)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        n_starts = 10
        starts = []

        # Add some structured starts
        # Center
        starts.append(np.zeros(dim))
        # Corners of [-4, 4]^5 (where minimizer lies)
        for i in range(1, 5):
            x = rng.uniform(-4, 4, dim)
            starts.append(x)

        # Random points
        while len(starts) < n_starts:
            x = rng.uniform(-5, 5, dim)
            starts.append(x)

        starts = np.array(starts[:n_starts])

        # Evaluate all starting points
        for s in starts:
            if time.time() > deadline:
                break
            evaluate(s)

        # For each start, do L-BFGS-B then Nelder-Mead refinement
        # Sort starts by initial objective value
        start_values = []
        for s in starts:
            if best_x is not None and np.allclose(s, best_x):
                continue
            start_values.append(s)

        # Process starts in order of their initial f (we already have best_f)
        # Do local refinement from the best few starts
        # We'll do L-BFGS-B from each start, then NM polish on the best results

        local_results = []

        for s in starts:
            if time.time() > deadline:
                break
            if call_count >= budget - 5:  # reserve some calls
                break

            # L-BFGS-B phase
            try:
                res = minimize(
                    evaluate,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 30}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                local_results.append(res.x.copy())
            except Exception:
                pass

            # Nelder-Mead polish on the best point found so far from this start
            if time.time() > deadline:
                break
            if call_count >= budget - 3:
                break

            # Use the best point as starting point for NM
            nm_start = best_x.copy() if best_x is not None else s
            try:
                res_nm = minimize(
                    evaluate,
                    nm_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': 20,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass

        # Final high-precision polish if time and budget remain
        if time.time() < deadline - 10 and call_count < budget - 2 and best_x is not None:
            try:
                res_final = minimize(
                    evaluate,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': 20,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
            except Exception:
                pass

        # Ensure at least one evaluation was made
        if call_count == 0:
            evaluate(np.zeros(dim))

        return best_x, best_f
# EVOLVE-BLOCK-END

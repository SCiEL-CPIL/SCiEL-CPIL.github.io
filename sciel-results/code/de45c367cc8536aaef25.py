"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + Nelder-Mead polish with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 250.0  # 250s safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Wrapper to count calls and clip
        call_count = [0]
        bounds = [(-5.0, 5.0)] * dim

        def wrapped(x):
            call_count[0] += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            return float(val)

        # Generate diverse starting points
        # Mix of random, grid-like, and zero-centered starts
        starts = []
        # Zero start
        starts.append(np.zeros(dim))
        # Random starts
        for _ in range(8):
            starts.append(rng.uniform(-5, 5, dim))
        # Some structured starts
        for _ in range(4):
            s = rng.choice([-4, -2, 0, 2, 4], size=dim)
            starts.append(s.astype(float))

        best_x = starts[0]
        best_f = wrapped(best_x)
        call_count[0] = 0  # Reset for actual optimization budget tracking

        # Re-evaluate to get proper count
        call_count[0] = 0

        # First, do a quick evaluation of all starts
        start_results = []
        for s in starts:
            if time.time() > deadline:
                break
            fs = wrapped(s)
            start_results.append((fs, s))
            if fs < best_f:
                best_f = fs
                best_x = s

        # Sort by function value, best first
        start_results.sort(key=lambda t: t[0])

        # Now do L-BFGS-B from each start, budget-aware
        # We have budget calls total. Each L-BFGS-B call uses multiple function evals.
        # Allocate budget across starts: more to better starts.
        # Simple approach: give each start a portion of remaining budget.

        remaining_budget = budget - call_count[0]
        n_starts = len(start_results)

        # Allocate: first start gets 40%, next 25%, rest split
        if n_starts == 0:
            return best_x, best_f

        allocations = []
        if n_starts >= 1:
            allocations.append(int(remaining_budget * 0.35))
        if n_starts >= 2:
            allocations.append(int(remaining_budget * 0.25))
        if n_starts >= 3:
            allocations.append(int(remaining_budget * 0.15))
        rest = remaining_budget - sum(allocations)
        if n_starts > 3:
            per_rest = max(1, rest // (n_starts - 3))
            allocations.extend([per_rest] * (n_starts - 3))
        elif rest > 0:
            allocations[-1] += rest

        # Ensure minimum 10 calls per start if possible
        for i in range(len(allocations)):
            allocations[i] = max(allocations[i], 10)

        for i, (fs, sx) in enumerate(start_results):
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break
            if i >= len(allocations):
                break

            max_iter = allocations[i]

            # L-BFGS-B
            try:
                res = minimize(
                    wrapped,
                    sx,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'maxfun': max_iter, 'gtol': 1e-6}
                )
            except Exception:
                res = None

            if res is not None and res.fun < best_f:
                best_f = res.fun
                best_x = res.x

            # Nelder-Mead polish from the best point found so far
            if time.time() > deadline:
                break
            if call_count[0] < budget:
                nm_budget = min(50, budget - call_count[0])
                if nm_budget > 10:
                    try:
                        nm_res = minimize(
                            wrapped,
                            best_x,
                            method='Nelder-Mead',
                            options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-8}
                        )
                        if nm_res.fun < best_f:
                            best_f = nm_res.fun
                            best_x = nm_res.x
                    except Exception:
                        pass

            # If we have budget left, do another round of L-BFGS-B from the current best
            if time.time() > deadline:
                break
            if call_count[0] < budget and i == 0:
                # Polish with more L-BFGS-B from the best found
                extra_budget = min(100, budget - call_count[0])
                if extra_budget > 20:
                    try:
                        res2 = minimize(
                            wrapped,
                            best_x,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': extra_budget, 'maxfun': extra_budget, 'gtol': 1e-8}
                        )
                        if res2.fun < best_f:
                            best_f = res2.fun
                            best_x = res2.x
                    except Exception:
                        pass

        # Final check
        if call_count[0] < budget and time.time() < deadline:
            # Try one more Nelder-Mead from best
            try:
                nm_final = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(30, budget - call_count[0]), 'xatol': 1e-9, 'fatol': 1e-9}
                )
                if nm_final.fun < best_f:
                    best_f = nm_final.fun
                    best_x = nm_final.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

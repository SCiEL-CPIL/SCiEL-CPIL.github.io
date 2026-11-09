"""Rewritten optimizer: deterministic coarse-to-fine 1D search + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Coarse-to-fine 1D grid search exploiting Rastrigin structure, then Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        # Track best solution and value
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse-to-fine 1D grid search
        # Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The global minimum is at x_i = k (integer) for the shifted/rotated version
        # We do a coarse grid first, then refine around the best point

        # Start with a coarse grid
        # For dim=5, we need to be efficient
        # Strategy: coordinate descent with 1D scans

        # Initialize at origin (common for shifted Rastrigin)
        x = np.zeros(self.dim)
        fx = safe_call(x)

        # Coarse-to-fine 1D coordinate search
        # Grid sizes: start coarse, get finer
        grid_sizes = [10, 20, 40, 80]

        for gs in grid_sizes:
            if calls_made >= self.budget or time.time() > deadline:
                break

            improved = True
            while improved and calls_made < self.budget and time.time() < deadline:
                improved = False
                for i in range(self.dim):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    # 1D scan along dimension i
                    # Scan range: [x[i] - 5, x[i] + 5] clipped to [-5, 5]
                    lo = max(-5.0, x[i] - 5.0)
                    hi = min(5.0, x[i] + 5.0)
                    grid = np.linspace(lo, hi, gs)
                    best_val = fx
                    best_coord = x[i]
                    for g in grid:
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        test_x = x.copy()
                        test_x[i] = g
                        tv = safe_call(test_x)
                        if tv < best_val:
                            best_val = tv
                            best_coord = g
                    if best_coord != x[i]:
                        x[i] = best_coord
                        fx = best_val
                        improved = True

        # Phase 2: Nelder-Mead polish from best point found
        if calls_made < self.budget and time.time() < deadline:
            # Use scipy's Nelder-Mead
            try:
                # Count how many calls we've used
                remaining = self.budget - calls_made
                # Nelder-Mead with maxiter
                max_iter = min(remaining // 2, 200)
                if max_iter > 0:
                    result = minimize(
                        func,
                        x0=best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    # Check if result is better
                    if result.fun < best_f:
                        best_x = result.x
                        best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

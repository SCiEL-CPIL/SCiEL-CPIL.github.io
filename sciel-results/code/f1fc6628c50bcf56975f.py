"""Optimized BBOB f9 (Rosenbrock rotated) solver with L-BFGS-B and restarts."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B for BBOB f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill

        # Wrap func to count calls and ensure we stay within budget
        state = {'count': 0, 'best_x': None, 'best_f': np.inf}

        def counted_func(x):
            state['count'] += 1
            if state['count'] > budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # Strategy: multiple restarts with L-BFGS-B
        # For Rosenbrock, good initial points near the valley help
        # The rotated Rosenbrock minimum is in [-4,4]^5

        # Phase 1: Try several random starts with L-BFGS-B
        n_restarts = min(8, max(3, budget // 60))
        calls_per_restart = budget // (n_restarts + 1)

        # Also try some structured starts
        start_points = []

        # Center and random points
        start_points.append(np.zeros(dim))
        start_points.append(rng.uniform(-3, 3, dim))
        start_points.append(rng.uniform(-4, 4, dim))

        # Points near typical Rosenbrock valley: x[i] near 1 in original,
        # but rotated so we just sample broadly
        for _ in range(4):
            start_points.append(rng.uniform(-5, 5, dim))

        # Sort by a heuristic: prefer points closer to center (since min is in [-4,4])
        def heuristic_dist(x):
            # Penalize points far from origin
            return np.sum((x - 0.0) ** 2)

        start_points.sort(key=heuristic_dist)

        best_x = None
        best_f = np.inf
        total_calls = 0

        for start in start_points:
            if total_calls >= budget - 10:
                break
            if time.time() > deadline:
                break

            x0 = np.clip(start, -5, 5).copy()
            # Evaluate starting point
            try:
                f0 = counted_func(x0)
                total_calls = state['count']
            except ValueError:
                break

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if total_calls >= budget - 10:
                break

            # Run L-BFGS-B from this start
            try:
                result = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfun': max(5, budget - total_calls - 5),
                             'ftol': 1e-12, 'gtol': 1e-8}
                )
                total_calls = state['count']

                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()

            except (ValueError, Exception):
                # Budget exceeded or other error - keep what we have
                break

            if total_calls >= budget:
                break

        # If we still have budget, do a local refinement with a tighter method
        if best_x is not None and total_calls < budget - 20 and time.time() < deadline:
            try:
                result2 = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': max(5, budget - total_calls - 5),
                             'ftol': 1e-15, 'gtol': 1e-10}
                )
                total_calls = state['count']
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = np.clip(result2.x, -5, 5).copy()
            except (ValueError, Exception):
                pass

        # Final: ensure best_x is set
        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END

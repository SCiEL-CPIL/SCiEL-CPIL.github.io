"""Rewritten optimizer: multi-start with Nelder-Mead polishing for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start local optimizer with Nelder-Mead for Rosenbrock-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 20s safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_made += 1
            return val

        # Strategy: multi-start with Nelder-Mead local polishing
        # Generate diverse starting points, then polish each with Nelder-Mead

        # Starting points: center, corners, and random
        rng = np.random.default_rng(42)

        # Build a list of candidate starting points
        starts = []

        # Center
        starts.append(np.zeros(dim))

        # Random points spread across the domain
        n_random = 30
        for _ in range(n_random):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Some structured points near boundaries
        for _ in range(10):
            x = rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0], size=dim)
            starts.append(x)

        # Sort starts by a heuristic: prefer center first, then random
        # Actually, let's just try them in order but interleave with local search

        # We'll do multi-start: for each start, run a short Nelder-Mead
        # Allocate budget across starts adaptively

        # Phase 1: Quick screening with a few function evaluations per start
        # Phase 2: Deep polishing on the best starts

        # Let's do a simpler approach:
        # 1. Evaluate all starts (dim+1 evals each for NM init)
        # 2. Track the best
        # 3. Continue polishing the best candidates

        # Actually, let's do sequential multi-start with adaptive budget allocation

        # Shuffle starts to get diversity early
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        # For each start, run Nelder-Mead with a limited budget
        # Total budget: 500 evaluations
        # We'll allocate ~15-20 evals per start initially, then more to good ones

        # Simple approach: run NM with maxiter based on remaining budget
        # Each NM iteration uses ~dim+1 evaluations

        n_starts = len(starts)
        # Give each start a small initial budget, then reallocate
        initial_budget_per_start = max(10, budget // (n_starts * 3))

        for i, x0 in enumerate(starts):
            if calls_made >= budget - 5:
                break
            if time.time() > deadline - 30:
                break

            # Run Nelder-Mead from x0
            # Use a small initial simplex
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = safe_call(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Remaining budget for this local search
            remaining = budget - calls_made - 5
            if remaining < 10:
                break

            # Allocate a portion of remaining budget
            # Diminishing returns: earlier starts get more budget
            frac = 0.3  # use 30% of remaining for this start
            local_budget = max(10, int(remaining * frac))

            try:
                result = minimize(
                    lambda x: safe_call(x),
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': local_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

            if result.success or result.fun < best_f:
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)

        # Phase 2: If we have budget left, do a final deep polish on the best point
        if best_x is not None and calls_made < budget - 10:
            remaining = budget - calls_made - 5
            if remaining > 20:
                try:
                    result = minimize(
                        lambda x: safe_call(x),
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
                except Exception:
                    pass

        # Ensure we don't exceed budget
        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END

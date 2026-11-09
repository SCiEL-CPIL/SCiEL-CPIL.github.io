"""Nelder-Mead with multi-start and restart for Rastrigin f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for box-constrained minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution globally
        best_x = None
        best_f = np.inf
        calls_used = 0

        # Generate diverse starting points
        rng = np.random.default_rng(42)

        # Mix of strategies for initial points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Corners (subset)
        for _ in range(8):
            starts.append(rng.choice([-4, -2, 0, 2, 4], size=dim))
        # Random uniform
        for _ in range(12):
            starts.append(rng.uniform(-4.5, 4.5, size=dim))
        # Small perturbations around center
        for _ in range(4):
            starts.append(rng.uniform(-0.5, 0.5, size=dim))

        # Sort by a heuristic: prefer points closer to center first (Rastrigin often has global min near shifted center)
        # Actually, let's just shuffle and use them in order, but evaluate the first few quickly
        # We'll use a budget allocation strategy:
        # Phase 1: Quick scan of all starts with a few NM iterations each
        # Phase 2: Deep dive on the best few

        # For AOCC, early progress matters a lot. So we want to find good values fast.
        # Strategy: Interleave evaluations from multiple starts to get early improvements.

        # Let's use a simpler approach:
        # 1. Evaluate all starts (len(starts) calls)
        # 2. Pick top K starts
        # 3. Run Nelder-Mead on each top start with remaining budget

        n_starts = len(starts)
        if n_starts >= budget:
            # Not enough budget for all starts, just do random search
            for _ in range(budget):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, size=dim)
                f = func(x)
                calls_used += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            return best_x, best_f

        # Phase 1: Evaluate all starts
        start_results = []
        for i, s in enumerate(starts):
            if time.time() > deadline:
                break
            x = np.clip(s, -5, 5)
            f = func(x)
            calls_used += 1
            start_results.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if not start_results:
            return best_x, best_f

        # Sort by fitness
        start_results.sort(key=lambda t: t[0])

        # Phase 2: Run Nelder-Mead on top starts
        # Allocate budget based on rank
        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Top K starts get NM refinement
        K = min(len(start_results), 5)
        # Budget allocation: more budget for better starts
        # e.g., rank 0 gets 40%, rank 1 gets 25%, rank 2 gets 15%, rank 3 gets 10%, rank 4 gets 10%
        weights = [0.40, 0.25, 0.15, 0.10, 0.10]
        if K < 5:
            # Redistribute
            weights = [w / sum(weights[:K]) for w in weights[:K]]

        for rank in range(K):
            if time.time() > deadline:
                break
            if remaining_budget <= 10:
                break

            f_init, x_init = start_results[rank]
            alloc = max(10, int(remaining_budget * weights[rank]))
            remaining_budget -= alloc

            # Run Nelder-Mead
            x_best, f_best = self._nelder_mead(
                func, x_init, f_init, alloc, dim, deadline, rng
            )
            calls_used += alloc  # Approximate; _nelder_mead tracks internally but let's be safe

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Phase 3: Final polish with small steps if budget remains
        if best_x is not None and remaining_budget > 5:
            sigma = 0.01
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.7

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, dim, deadline, rng):
        """Nelder-Mead simplex method with box constraints and adaptive restarts."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [f0]
        calls = 1

        # Generate other simplex vertices
        for i in range(dim):
            x_new = x0.copy()
            if x0[i] != 0:
                x_new[i] += 0.1 * abs(x0[i])
            else:
                x_new[i] += 0.1
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls += 1
            simplex.append(x_new)
            fvals.append(f_new)
            if f_new < f0:
                f0 = f_new
                x0 = x_new.copy()

        # Nelder-Mead iteration
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        stagnation = 0
        max_stagnation = 10

        for _ in range(budget - calls):
            if time.time() > deadline:
                break

            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check for convergence
            if max(fvals) - min(fvals) < 1e-10:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                    stagnation = 0
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    stagnation = 0
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
                stagnation = 0
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                    stagnation = 0
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                    stagnation += 1

            if stagnation > max_stagnation:
                break

        # Return best in simplex
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END

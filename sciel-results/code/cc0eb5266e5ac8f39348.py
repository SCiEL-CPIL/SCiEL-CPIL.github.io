"""Hybrid optimizer: LHS global search + Nelder-Mead local polish for f24."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Latin Hypercube global search + Nelder-Mead local exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget_remaining, deadline, n, best_state):
        """Nelder-Mead simplex method with adaptive step sizes."""
        if budget_remaining <= 0 or time.time() >= deadline:
            return best_state

        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]
        budget_remaining -= 1
        if fvals[0] < best_state[1]:
            best_state = (x0.copy(), fvals[0])

        # Build initial simplex with small perturbations
        for i in range(n):
            xi = x0.copy()
            step = 0.1 if abs(x0[i]) > 1e-8 else 0.1
            xi[i] += step
            xi = np.clip(xi, -5.0, 5.0)
            simplex.append(xi)
            fvals.append(func(xi))
            budget_remaining -= 1
            if fvals[-1] < best_state[1]:
                best_state = (xi.copy(), fvals[-1])
            if budget_remaining <= 0 or time.time() >= deadline:
                break

        if len(simplex) < 2:
            return best_state

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        max_iter = max(1, budget_remaining // (n + 1))

        for _ in range(max_iter):
            if budget_remaining <= 0 or time.time() >= deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_state[1]:
                best_state = (simplex[0].copy(), fvals[0])

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)
            budget_remaining -= 1
            if f_r < best_state[1]:
                best_state = (x_r.copy(), f_r)

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                budget_remaining -= 1
                if f_e < best_state[1]:
                    best_state = (x_e.copy(), f_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Simple reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = func(x_c)
                budget_remaining -= 1
                if f_c < best_state[1]:
                    best_state = (x_c.copy(), f_c)

                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = func(simplex[i])
                        budget_remaining -= 1
                        if fvals[i] < best_state[1]:
                            best_state = (simplex[i].copy(), fvals[i])
                        if budget_remaining <= 0 or time.time() >= deadline:
                            break
                    if budget_remaining <= 0 or time.time() >= deadline:
                        break

        return best_state

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Dense Latin Hypercube Sampling for global basin identification
        # Use ~60% of budget for exploration
        n_lhs = min(int(budget * 0.6), 300)
        n_lhs = max(n_lhs, 50)  # minimum samples

        rng = np.random.default_rng(42)
        lhs_points = np.zeros((n_lhs, n))
        for d in range(n):
            # Latin hypercube: divide [0,1] into n_lhs equal intervals
            # Random permutation of intervals for each dimension
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = (perm + rng.random(n_lhs)) / n_lhs
            # Map to [-5, 5]
            lhs_points[:, d] = -5.0 + 10.0 * lhs_points[:, d]

        # Evaluate all LHS points
        lhs_fvals = np.empty(n_lhs)
        for i in range(n_lhs):
            if time.time() >= deadline:
                break
            x = np.clip(lhs_points[i], -5.0, 5.0)
            f = func(x)
            lhs_fvals[i] = f
            budget -= 1
            update_best(x, f)
            if budget <= 0:
                break

        # Identify top-k candidates from LHS
        k = min(5, n_lhs)
        top_idx = np.argsort(lhs_fvals)[:k]

        # Phase 2: Nelder-Mead local optimization from top candidates
        # Distribute remaining budget among top candidates
        remaining_budget = budget
        if remaining_budget > 0 and time.time() < deadline:
            # Allocate budget: give more to better candidates
            for rank, idx in enumerate(top_idx):
                if remaining_budget <= 0 or time.time() >= deadline:
                    break
                # Allocate decreasing budget to each candidate
                alloc = max(10, remaining_budget // (len(top_idx) - rank))
                alloc = min(alloc, remaining_budget)
                x0 = np.clip(lhs_points[idx], -5.0, 5.0)
                best_state = (x0.copy(), lhs_fvals[idx])
                best_state = self._nelder_mead(
                    func, x0, alloc, deadline, n, best_state
                )
                update_best(best_state[0], best_state[1])
                remaining_budget -= alloc

        # Phase 3: If budget remains, do a fine-grained local search around best point
        if budget > 10 and time.time() < deadline and np.isfinite(best_f):
            # Small random perturbations around best point
            n_fine = min(budget, 50)
            for i in range(n_fine):
                if budget <= 0 or time.time() >= deadline:
                    break
                # Adaptive step size based on current best
                step = max(0.01, 0.1 * np.exp(-best_f / 50.0))
                x = best_x + step * rng.standard_normal(n)
                x = np.clip(x, -5.0, 5.0)
                f = func(x)
                budget -= 1
                update_best(x, f)

        return best_x, best_f
# EVOLVE-BLOCK-END

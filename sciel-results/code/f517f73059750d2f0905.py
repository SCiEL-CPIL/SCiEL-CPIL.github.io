"""Structure-aware lattice search for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware lattice search for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        call_count = [0]
        best_x = [None]
        best_f = [float('inf')]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > self.budget:
                return best_f[0]
            fx = func(np.clip(x, -5.0, 5.0))
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        # Katsuura function has a known global minimum at the origin in the
        # canonical (unshifted) space. The hidden transformation is:
        #   f(x) = g(R @ (x - s) + o) + c
        # where g is the canonical Katsuura, R is rotation, s is shift, o is offset, c is constant.
        #
        # Strategy: We don't know s, R, o, c. But we can do a multi-resolution
        # search that starts coarse and refines around the best point found.
        # The key insight: Katsuura is highly multimodal with oscillations that
        # increase in frequency near the optimum. A coarse grid will miss the
        # global minimum, but a hierarchical refinement can find it.
        #
        # We use a deterministic Halton sequence for space-filling, combined
        # with local refinement around the best point found so far.

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse space-filling search using Halton sequence
        # Evaluate ~40% of budget on a coarse Halton grid
        n_coarse = max(20, int(budget * 0.4))

        def halton(index, base, dim):
            """Generate Halton sequence point."""
            f = np.zeros(dim)
            for d in range(dim):
                base_d = base + d
                result = 0.0
                f = 1.0
                i = index
                while i > 0:
                    i //= base_d
                    f /= base_d
                    result += f * (i % base_d)
                f[d] = result
            return f

        # Evaluate Halton points
        for i in range(1, n_coarse + 1):
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            x = halton(i, 2, dim) * 10.0 - 5.0  # Map to [-5, 5]^dim
            counted_func(x)

        # Phase 2: Local refinement around the best point found
        # Use a multi-scale local search with decreasing step sizes
        # This exploits the fact that Katsuura has a sharp minimum

        remaining = budget - call_count[0]
        if remaining < 5 or best_x[0] is None:
            return best_x[0], best_f[0]

        # Multi-scale refinement: try step sizes from large to small
        # For each scale, do a coordinate-wise local search
        scales = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

        # Allocate remaining budget across scales
        n_scales = len(scales)
        # Give more budget to earlier (coarser) scales
        weights = np.array([3, 2.5, 2, 1.5, 1.2, 1, 0.8, 0.6, 0.5])
        weights = weights / weights.sum()
        scale_budgets = (remaining * weights).astype(int)
        scale_budgets = np.maximum(scale_budgets, 2)  # At least 2 calls per scale

        for scale_idx, step_size in enumerate(scales):
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            scale_budget = scale_budgets[scale_idx]
            if scale_budget < 2:
                continue

            # Coordinate-wise local search: try moving in +/- direction for each dimension
            # Plus diagonal moves for better exploration
            x_current = best_x[0].copy()
            f_current = best_f[0]

            calls_this_scale = 0

            # Try coordinate moves
            for coord in range(dim):
                for direction in [1, -1]:
                    if call_count[0] >= budget or calls_this_scale >= scale_budget:
                        break
                    x_trial = x_current.copy()
                    x_trial[coord] += direction * step_size
                    x_trial = np.clip(x_trial, -5.0, 5.0)

                    # Don't re-evaluate if we've already tried this point
                    fx = counted_func(x_trial)
                    calls_this_scale += 1

                    if fx < f_current:
                        f_current = fx
                        x_current = x_trial.copy()

            # Try diagonal moves (move all coordinates by small step)
            if calls_this_scale < scale_budget and call_count[0] < budget:
                for direction in [1, -1]:
                    if call_count[0] >= budget or calls_this_scale >= scale_budget:
                        break
                    x_trial = x_current + direction * step_size * 0.5 * np.ones(dim)
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    fx = counted_func(x_trial)
                    calls_this_scale += 1
                    if fx < f_current:
                        f_current = fx
                        x_current = x_trial.copy()

            # Update global best if local search found something better
            if f_current < best_f[0]:
                best_f[0] = f_current
                best_x[0] = x_current.copy()

        # Phase 3: Final fine-tuning with very small steps if budget remains
        remaining = budget - call_count[0]
        if remaining >= 5 and best_x[0] is not None and time.time() < deadline - 5.0:
            # Very fine coordinate search
            fine_step = 0.001
            for coord in range(dim):
                for direction in [1, -1]:
                    if call_count[0] >= budget or time.time() > deadline - 5.0:
                        break
                    x_trial = best_x[0].copy()
                    x_trial[coord] += direction * fine_step
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    fx = counted_func(x_trial)
                    if fx < best_f[0]:
                        best_f[0] = fx
                        best_x[0] = x_trial.copy()

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END

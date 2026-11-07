"""Lattice search exploiting Rastrigin's integer-grid structure with adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice search for Rastrigin f3 with adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0

        def call(x):
            nonlocal calls
            x = np.clip(x, -5, 5)
            calls += 1
            return float(func(x))

        # Phase 1: Lattice scan at step 0.5 (exploits Rastrigin periodicity)
        # Domain [-4, 4]^5 since minimizer is in [-4, 4]^5
        # Step 0.5 gives 17^5 = 1419857 points - too many for 500 budget
        # Step 1.0 gives 9^5 = 59049 - still too many
        # We need to be selective. Use a stratified approach.

        # Since budget is 500, we need a smart sampling strategy.
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global min at x = 0 (shifted). Local minima at integer coordinates.
        # The shift is hidden but minimizer is in [-4,4]^5.

        # Strategy: Multi-resolution lattice search
        # Level 1: Coarse grid step 2.0 -> points at -4,-2,0,2,4 in each dim -> 5^5 = 3125
        # Level 2: Medium grid step 1.0 -> 9^5 = 59049 (too many)
        # Level 3: Fine grid step 0.5 around best coarse point

        # With 500 budget, let's do:
        # - Sample ~200 points on a stratified coarse grid
        # - Refine around the best ~100 points
        # - Final local search with ~100 points

        # Actually, let's think more carefully.
        # The key insight: Rastrigin has minima at integer coordinates.
        # After shift, the global minimum is at some integer-like coordinate.
        # A lattice at step 1.0 would hit all integer points.
        # 9^5 = 59049 is too many. But we can use a space-filling curve or
        # stratified sampling to get good coverage with 500 points.

        # Better approach: Use a multi-stage lattice:
        # Stage 1: 5^5 = 3125 points at step 2.0 -> sample 150 of these
        # Stage 2: 3^5 = 243 points at step 0.5 around best stage-1 point
        # Stage 3: 5^5 = 3125 points at step 0.2 around best stage-2 point -> sample 100
        # Stage 4: Local refinement with 100 calls

        # Actually, with only 500 budget, let's be more aggressive:
        # The rotation makes this harder. Let's use a combination:
        # 1. Random + lattice hybrid for initial exploration
        # 2. Adaptive refinement

        # Let me reconsider. The rotation is the key challenge.
        # Lattice search works if the function is axis-aligned, but rotation breaks this.
        # However, the Rastrigin structure (periodic in each dimension) means that
        # even with rotation, the landscape has a grid-like structure in the
        # rotated coordinate system.

        # Best strategy given constraints:
        # 1. Evaluate on a stratified grid to find promising regions
        # 2. Use adaptive step-size local search around the best points
        # 3. Multi-restart to handle multiple local minima

        # Phase 1: Stratified sampling (200 points)
        # Use a Halton-like sequence for good space-filling
        n_phase1 = 150
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Generate stratified points: divide each dim into 5 cells, pick random point in each
        # 5^5 = 3125 cells, sample 150
        cells = list(np.ndindex([5] * dim))
        rng.shuffle(cells)
        for cell in cells[:n_phase1]:
            if calls >= budget:
                break
            x = np.array([(cell[i] + rng.random()) * 2.0 - 4.0 for i in range(dim)])
            fx = call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Refine around best point with step 1.0 (100 points)
        n_phase2 = 100
        if best_x is not None and calls < budget:
            step2 = 1.0
            offsets = list(np.ndindex([3] * dim))  # -1, 0, 1 in each dim
            rng.shuffle(offsets)
            for off in offsets:
                if calls >= budget:
                    break
                x = best_x + (np.array(off) - 1) * step2
                if np.any(x < -5) or np.any(x > 5):
                    continue
                fx = call(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Finer refinement around best (step 0.5, 100 points)
        n_phase3 = 100
        if best_x is not None and calls < budget:
            step3 = 0.5
            offsets = list(np.ndindex([5] * dim))  # -2,-1,0,1,2 in each dim
            rng.shuffle(offsets)
            for off in offsets:
                if calls >= budget:
                    break
                x = best_x + (np.array(off) - 2) * step3
                if np.any(x < -5) or np.any(x > 5):
                    continue
                fx = call(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 4: Very fine local search (step 0.1, remaining budget)
        if best_x is not None and calls < budget:
            remaining = budget - calls
            # Use a simple (1+1)-ES with small step for final polish
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 5: If time and budget remain, try a few more restarts
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                # Restart near best with perturbation
                x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                fx = call(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                else:
                    # Local search from this point with small steps
                    for _ in range(3):
                        if calls >= budget or time.time() > deadline:
                            break
                        y = np.clip(x + 0.2 * rng.standard_normal(dim), -5, 5)
                        fy = call(y)
                        if fy < fx:
                            x, fx = y, fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END

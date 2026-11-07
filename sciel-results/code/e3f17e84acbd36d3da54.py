"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with Golden Section Search optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: Random in [-4, 4] to handle shifts
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Intervals for each dimension
        # Start wide to ensure we don't miss the optimum if shift is large
        # But [-5, 5] is the domain.
        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)

        # Golden section ratio
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        gr = 1.0 / phi  # ~0.618
        gr2 = 1.0 - gr  # ~0.382

        # We will iterate through dimensions repeatedly.
        # In each pass for dimension d, we perform Golden Section Search.
        # GSS maintains two interior points.
        # We need to be careful with budget.

        # To maximize efficiency, we do a "round" of CD.
        # In each round, for each dim, we do a few GSS steps.
        # Or, we can just keep doing GSS on the "current" dim until interval is small, then move to next.
        # Given the anisotropy, it's good to cycle through dimensions.
        
        # Let's use a simple loop:
        # While calls < budget:
        #   For d in range(dim):
        #     Do 1-2 GSS steps on dim d.
        #     Update x, fx, best.
        #     Shrink lo/hi.

        # To avoid getting stuck in a bad local minimum (though Bent Cigar is convex),
        # random restarts or large initial steps help.
        # Since it's convex, CD should converge to the global min.

        while calls < budget:
            if time.time() > deadline:
                break

            for d in range(dim):
                if calls >= budget or time.time() > deadline:
                    break

                # Perform Golden Section Search on dimension d
                # Current interval [lo[d], hi[d]]
                # We maintain the invariant that the minimum is within [lo[d], hi[d]]
                
                # To save calls, we can do multiple GSS iterations per dimension visit
                # if the interval is still wide.
                # Let's do at least 1 GSS step.
                
                # Initialize GSS points if this is the first time or if we reset
                # Actually, standard GSS keeps state. But here we just use the current x as a point inside.
                # Simpler: Just do a binary-search-like GSS from scratch every time? No, that wastes calls.
                
                # Better: Keep track of the last two points and their values for each dimension?
                # That's complex. 
                
                # Alternative: Just use a simple "move" strategy combined with GSS refinement.
                # But GSS is better.
                
                # Let's implement a stateless GSS that takes the current best x as the center of interest.
                # Actually, if we just shrink the interval around the best x found so far, we can use a simplified GSS.
                
                # Let's try this:
                # 1. Evaluate at two points inside [lo[d], hi[d]] that are symmetric around current best or just standard GSS points.
                # Standard GSS:
                #   a = lo[d], b = hi[d]
                #   c = b - (b-a)*gr
                #   d_pt = a + (b-a)*gr
                #   Evaluate f(c), f(d_pt)
                #   If f(c) < f(d_pt): b = d_pt, else a = c
                #   Move the better point to the new position.
                
                # To do this efficiently, we need to store the function values of the interior points.
                # Let's store them in arrays.
                
                # We'll do this:
                # For each dimension, we maintain a small history or just re-evaluate?
                # Re-evaluating is expensive.
                
                # Let's use a simpler robust approach:
                # "Line Search" via GSS, but we reset the GSS state every time we switch dimensions?
                # No, that wastes the previous narrowing.
                
                # Let's keep `gss_c`, `gss_d`, `f_c`, `f_d` for each dimension.
                # But x changes when we optimize other dimensions, so f_c and f_d become stale!
                # When we change x[other_dim], the function value at fixed x[d] changes.
                # So we MUST re-evaluate or invalidate the GSS state.
                
                # This implies that standard GSS state cannot be kept across dimension switches in CD.
                # So, for each dimension visit, we must start a fresh GSS or a very limited one.
                
                # If we start fresh GSS every time, we waste calls re-evaluating points we already know.
                
                # Alternative: 
                # Use a "Narrowing Interval" approach.
                # 1. Pick a point y in [lo[d], hi[d]] (e.g., current x[d] or midpoint).
                # 2. Evaluate.
                # 3. If better, move x[d] to y.
                # 4. Shrink [lo[d], hi[d]] to [lo[d], y] or [y, hi[d]]? No, we don't know which side is better without 2 points.
                
                # Let's use the 2-point GSS logic but accept that we re-evaluate.
                # To minimize waste, we can choose the points close to the current best x[d].
                
                # Strategy:
                # For dim d:
                #   If interval is large, do 1 GSS step (2 evals).
                #   If interval is small, do 1 GSS step (2 evals) to refine.
                #   Update x[d] to the best of the two.
                #   Update lo[d], hi[d] based on GSS logic.
                
                # We need to store the "other" GSS point value from the previous step for THIS dimension?
                # No, because x[other] changed.
                
                # So, we effectively do a fresh 2-point comparison each time.
                # This is less efficient than full GSS but robust.
                
                # Let's try to be smarter:
                # Keep x[d] as the "best known" for that dim.
                # Pick a step size based on current interval width.
                # Try x[d] + step and x[d] - step?
                # Or just standard GSS on [lo, hi].
                
                # Let's stick to standard GSS on [lo[d], hi[d]].
                # We will perform 1 iteration of GSS (2 function calls) per dimension visit.
                # This shrinks the interval by factor ~0.618 each visit.
                # 500 calls / 5 dims = 100 visits per dim.
                # 0.618^100 is extremely small. Convergence is guaranteed.
                
                a = lo[d]
                b = hi[d]
                
                # Generate two points
                c = b - (b - a) * gr
                d_pt = a + (b - a) * gr
                
                # Evaluate
                # We need to construct the full vectors
                x_c = x.copy()
                x_c[d] = c
                f_c = func(x_c)
                calls += 1
                if calls >= budget: break
                
                x_d = x.copy()
                x_d[d] = d_pt
                f_d = func(x_d)
                calls += 1
                if calls >= budget: break
                
                # Update best
                if f_c < best_fx:
                    best_fx = f_c
                    best_x = x_c.copy()
                    # Update x for next iterations
                    x[d] = c
                    fx = f_c
                elif f_d < best_fx:
                    best_fx = f_d
                    best_x = x_d.copy()
                    x[d] = d_pt
                    fx = f_d
                
                # Update interval
                if f_c < f_d:
                    # Min is in [a, d_pt]
                    hi[d] = d_pt
                    # We could store f_c and c for next time, but x changes...
                    # So we discard state.
                else:
                    # Min is in [c, b]
                    lo[d] = c

        return best_x, best_fx
# EVOLVE-BLOCK-END

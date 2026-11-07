"""Rewritten program: coordinate descent with golden-section search for Bent Cigar."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate descent with 1D golden-section search per dimension.

    The Bent Cigar function has a very narrow valley along one axis and broad
    curvature along the others.  By decoupling each coordinate and running a
    precise 1D search (golden-section) we can exploit this anisotropy directly.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        # Start from a random point inside the box
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)

        # Golden-section constant
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
        inv_gr = 1.0 / gr                 # ~1.618

        # Pre-compute how many 1D evaluations we can afford per full sweep.
        # Each dimension's golden-section search uses a fixed number of evals.
        # We'll do multiple sweeps until budget is exhausted.
        # For 5 dims, each golden-section search with ~15 iterations uses ~17 evals.
        # Total per sweep ~85 evals.  Budget 500 -> ~5 sweeps.

        # Strategy: alternate between a coarse random restart if stuck, and
        # coordinate descent sweeps.

        # We'll structure the loop as:
        #   - Perform one full coordinate-descent sweep (one golden-section per dim)
        #   - Repeat until budget runs out.

        # Each golden-section search on a 1D interval uses at most `gs_iters` evaluations.
        # Let's use 20 iterations per dimension per sweep -> 20*5 = 100 evals per sweep.
        # That gives us ~4-5 sweeps.

        gs_iters = 20  # iterations of golden-section per dimension per sweep

        calls_made = 1  # initial evaluation

        # Helper: perform golden-section search on dimension d, holding others fixed.
        # Returns new value for x[d] and the best fx found.
        def golden_section_search(x, d, lo, hi, fx, gs_iters, func, budget_remaining):
            """Perform golden-section search on dimension d.
            
            Returns (new_x_d, best_fx, evals_used).
            """
            nonlocal calls_made
            
            # Initial bracket: use the current point and expand slightly
            # Actually, for golden-section we need a bracket [lo, hi].
            # We'll use the box bounds [-5, 5] as the initial bracket, but that's
            # too wide. Better: use a local interval around x[d].
            
            # Let's use a local interval: [x[d] - 1.0, x[d] + 1.0] clipped to [-5, 5]
            # But if the minimum is far, we might miss it. 
            # Alternative: start with the full box and narrow down.
            
            # For efficiency, let's use a moderate initial interval around current x[d].
            # If the function is unimodal along this axis (which Bent Cigar is,
            # since it's a sum of squares with different coefficients), golden-section
            # will work well.
            
            # Use interval [x[d] - 2.0, x[d] + 2.0] clipped to [-5, 5]
            half_width = 2.0
            lo = max(-5.0, x[d] - half_width)
            hi = min(5.0, x[d] + half_width)
            
            # If the interval is too small, expand
            if hi - lo < 1e-6:
                lo = max(-5.0, x[d] - 1.0)
                hi = min(5.0, x[d] + 1.0)
            
            # Golden-section search
            a = lo
            b = hi
            c = b - gr * (b - a)
            d_point = a + gr * (b - a)
            
            # Evaluate c and d
            x_c = x.copy()
            x_c[d] = c
            fc = func(x_c)
            calls_made += 1
            
            x_d = x.copy()
            x_d[d] = d_point
            fd = func(x_d)
            calls_made += 1
            
            # Keep the better of fc, fd as the current best
            if fc < fd:
                # Minimum is in [a, d_point]
                b = d_point
                d_point = c
                fd = fc
                c = b - gr * (b - a)
                x_c = x.copy()
                x_c[d] = c
                fc = func(x_c)
                calls_made += 1
            else:
                # Minimum is in [c, b]
                a = c
                c = d_point
                fc = fd
                d_point = a + gr * (b - a)
                x_d = x.copy()
                x_d[d] = d_point
                fd = func(x_d)
                calls_made += 1
            
            # Continue for remaining iterations
            for _ in range(gs_iters - 2):
                if fc < fd:
                    b = d_point
                    d_point = c
                    fd = fc
                    c = b - gr * (b - a)
                    x_c = x.copy()
                    x_c[d] = c
                    fc = func(x_c)
                    calls_made += 1
                else:
                    a = c
                    c = d_point
                    fc = fd
                    d_point = a + gr * (b - a)
                    x_d = x.copy()
                    x_d[d] = d_point
                    fd = func(x_d)
                    calls_made += 1
                
                # Check if we're out of budget
                if calls_made >= budget:
                    break
            
            # The best point is the one with lower function value
            if fc < fd:
                best_d = c
                best_fx = fc
            else:
                best_d = d_point
                best_fx = fd
            
            evals_used = 2 + (gs_iters - 2)  # approximate
            return best_d, best_fx, evals_used

        # Main loop: perform coordinate descent sweeps
        while calls_made < self.budget:
            improved = False
            for d in range(self.dim):
                if calls_made >= self.budget:
                    break
                
                # Perform golden-section search on dimension d
                new_d, new_fx, _ = golden_section_search(
                    x, d, -5.0, 5.0, fx, gs_iters, func, self.budget
                )
                
                if new_fx < fx:
                    x[d] = new_d
                    fx = new_fx
                    improved = True
            
            # If no improvement in a full sweep, try a small random perturbation
            # to escape potential local minima (though Bent Cigar is unimodal)
            if not improved and calls_made < self.budget:
                # Try a few random steps
                for _ in range(5):
                    if calls_made >= self.budget:
                        break
                    y = x + rng.normal(0, 0.1, self.dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        break
        
        return x, fx
# EVOLVE-BLOCK-END

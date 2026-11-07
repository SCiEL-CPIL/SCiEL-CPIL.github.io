# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Global Bayesian-style coordinate-descent optimizer for the Bent Cigar
    function.  Uses a cheap global exploration phase (Latin-hypercube sampling)
    followed by adaptive coordinate descent with dynamic range shrinking,
    exploiting the extreme anisotropy of the Bent Cigar valley."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                if f < best_f:
                    best_f = f
                    best_x = x_clipped.copy()
                return f
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # Phase 1: Global exploration with Latin-hypercube sampling
        # Use ~15% of budget for broad search to locate the valley region
        n_lh = max(10, min(50, int(budget * 0.15)))
        # Latin hypercube: one point per stratum in each dimension
        for i in range(n_lh):
            if calls >= budget - 5 or time.time() > deadline:
                break
            # Stratified sampling: divide each dimension into n_lh strata
            x = np.zeros(dim)
            for d in range(dim):
                stratum = (i + rng.random()) / n_lh
                x[d] = -5.0 + 10.0 * stratum
            wrapper(x)

        # Phase 2: Adaptive coordinate descent with dynamic range shrinking
        # The Bent Cigar has a narrow valley along one direction and wide
        # cross-sections.  We iteratively refine each coordinate using
        # golden-section search, shrinking the range as we converge.
        
        if calls >= budget - 10 or time.time() > deadline:
            return best_x, best_f

        # Start from the best point found so far
        x = best_x.copy()
        fx = best_f

        # Adaptive ranges: start wide, shrink based on progress
        # For each coordinate, maintain a search range [lo, hi]
        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)

        # Number of golden-section evaluations per coordinate per sweep
        # More evaluations per coordinate exploit the anisotropy better
        golden_evals_per_coord = 8

        # Maximum number of sweeps before we consider converged
        max_sweeps = 100
        min_range = 1e-4

        for sweep in range(max_sweeps):
            if calls >= budget - 5 or time.time() > deadline:
                break

            sweep_improved = False

            for d in range(dim):
                if calls >= budget - 5 or time.time() > deadline:
                    break

                # Current range for coordinate d
                lo_d = lo[d]
                hi_d = hi[d]
                range_d = hi_d - lo_d

                if range_d < min_range:
                    continue

                # Golden-section search on coordinate d
                # We perform golden_evals_per_coord evaluations
                # to narrow down the range
                invphi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
                invphi2 = invphi * invphi  # ~0.382

                # Initial bracket
                a = lo_d
                b = hi_d
                c = a + invphi2 * (b - a)
                d_pt = a + invphi * (b - a)

                # Evaluate at c and d
                x_c = x.copy()
                x_c[d] = c
                fc = wrapper(x_c)

                if calls >= budget - 5 or time.time() > deadline:
                    break

                x_d = x.copy()
                x_d[d] = d_pt
                fd = wrapper(x_d)

                # Iteratively narrow the bracket
                remaining_evals = golden_evals_per_coord - 2
                for _ in range(remaining_evals):
                    if calls >= budget - 5 or time.time() > deadline:
                        break

                    if fc < fd:
                        # Minimum is in [a, d]
                        b = d_pt
                        d_pt = c
                        fd = fc
                        c = a + invphi2 * (b - a)
                        x_c = x.copy()
                        x_c[d] = c
                        fc = wrapper(x_c)
                    else:
                        # Minimum is in [c, b]
                        a = c
                        c = d_pt
                        fc = fd
                        d_pt = a + invphi * (b - a)
                        x_d = x.copy()
                        x_d[d] = d_pt
                        fd = wrapper(x_d)

                # Update best point if we found something better
                best_in_bracket = min(fc, fd)
                best_pt = c if fc < fd else d_pt
                if best_in_bracket < fx:
                    x[d] = best_pt
                    fx = best_in_brand if best_in_brand < fx else best_in_brand
                    # Fix: use the actual value
                    fx = best_in_bracket
                    sweep_improved = True

                # Update the range for next sweep
                new_lo = max(a, lo_d)
                new_hi = min(b, hi_d)
                # Shrink range: keep the bracket but also shrink toward best
                mid = (a + b) / 2.0
                half_width = (b - a) / 2.0
                lo[d] = max(lo_d, mid - half_width)
                hi[d] = min(hi_d, mid + half_width)

            # If no improvement in this sweep, shrink all ranges aggressively
            if not sweep_improved:
                for d in range(dim):
                    mid = (lo[d] + hi[d]) / 2.0
                    half = (hi[d] - lo[d]) / 4.0
                    lo[d] = mid - half
                    hi[d] = mid + half

                # If ranges are too small, try a random restart in a wider area
                all_small = all((hi[d] - lo[d]) < 1e-2 for d in range(dim))
                if all_small and calls < budget - 20 and time.time() < deadline:
                    # Random restart: pick a new point and re-initialize ranges
                    x = best_x.copy()
                    fx = best_f
                    # Try a few random perturbations
                    for _ in range(5):
                        if calls >= budget - 5 or time.time() > deadline:
                            break
                        x_try = best_x + rng.normal(0, 1.0, dim)
                        x_try = np.clip(x_try, -5, 5)
                        ft = wrapper(x_try)
                        if ft < fx:
                            x = x_try
                            fx = ft
                    # Re-initialize ranges around current best
                    lo = x - 1.0
                    hi = x + 1.0
                    lo = np.clip(lo, -5, 5)
                    hi = np.clip(hi, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END

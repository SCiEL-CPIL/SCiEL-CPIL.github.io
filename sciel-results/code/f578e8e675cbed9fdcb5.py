"""Multi-objective optimizer for ZDT1: hypervolume-driven adaptive search on x1."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven optimizer exploiting ZDT1's 1-D Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # We will manage a set of candidate x1 values and evaluate them.
        # Keep a list of (x1, x_full, f) for the archive; only x1 matters for the front.
        archive_x1 = []
        archive_x = []
        archive_f = []
        used = 0

        def time_ok():
            return (time.time() - start) < deadline

        def add_candidate(x1, x_full, f):
            nonlocal used
            # Only keep if f is finite and strictly better than ref [11,11] in every obj
            if not (np.all(np.isfinite(f))):
                return
            if f[0] >= 11.0 or f[1] >= 11.0:
                return
            archive_x1.append(float(x1))
            archive_x.append(x_full)
            archive_f.append(np.asarray(f, dtype=float))
            used += 1

        def make_x(x1):
            # Build a full vector: x[0]=x1, rest random in [lo,hi]
            x = np.empty(dim)
            x[0] = x1
            if dim > 1:
                x[1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random(dim - 1)
            return x

        # Phase 1: dense uniform grid over x1 in [0,1] (the known front domain)
        # The front spans x1 in [0,1]. Sample a grid.
        n_grid = min(40, budget // 4)
        if n_grid < 2:
            n_grid = 2
        grid = np.linspace(0.0, 1.0, n_grid)
        for x1 in grid:
            if used >= budget or not time_ok():
                break
            x = make_x(x1)
            f = problem(x)
            add_candidate(x1, x, f)

        # Phase 2: adaptive bisection + random refinement to densify coverage
        # We want to maximize hypervolume, which for a line front means
        # dense coverage of x1 in [0,1]. Use iterative subdivision.
        # Maintain a set of intervals to refine, prioritizing largest gaps.
        # Simpler: repeatedly pick random x1 in [0,1] and also bisect largest gaps.
        # Track sorted x1 values to find gaps.

        # We'll do a mix: each iteration, with prob 0.5 bisect the largest gap,
        # else sample randomly.
        max_iter = budget - used
        it = 0
        while used < budget and it < max_iter and time_ok():
            it += 1
            # Get sorted archive x1
            if len(archive_x1) >= 2:
                sx = np.array(sorted(archive_x1))
                # gaps including endpoints 0 and 1
                # We consider the domain [0,1]; find largest gap between consecutive
                # points (and edges).
                ext = np.concatenate(([-1.0], sx, [2.0]))
                gaps = np.diff(ext)
                # Only consider gaps within [0,1] roughly; pick the largest interior gap
                # Find index of largest gap
                idx = int(np.argmax(gaps))
                # The gap idx is between ext[idx] and ext[idx+1]
                a = ext[idx]
                b = ext[idx + 1]
                # Clamp to [0,1]
                a = max(0.0, min(1.0, a))
                b = max(0.0, min(1.0, b))
                if b - a < 1e-4:
                    # gap too small, sample randomly instead
                    x1 = rng.random()
                else:
                    x1 = 0.5 * (a + b)
            else:
                x1 = rng.random()

            # Occasionally do a random sample to avoid over-focusing
            if rng.random() < 0.3:
                x1 = rng.random()

            x = make_x(x1)
            f = problem(x)
            add_candidate(x1, x, f)

        # Phase 3: local polish - for a few of the best hypervolume-contributing points,
        # do a fine local search in x1 to nail down exact front location.
        # Actually the front is known to be f1=x1, f2=1-x1, so any x1 in [0,1] is on the
        # front. The hypervolume only depends on the set of (f1,f2) = (x1,1-x1) points.
        # So we just need dense coverage. We already have that.
        # Do a final high-resolution pass if budget remains: sample many random x1.
        while used < budget and time_ok():
            x1 = rng.random()
            x = make_x(x1)
            f = problem(x)
            add_candidate(x1, x, f)

        return archive_x
# EVOLVE-BLOCK-END

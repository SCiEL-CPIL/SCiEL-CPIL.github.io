"""Rewritten program for WFG7 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume-guided local search on WFG7 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # WFG7 Pareto front: for 3 objectives, the front is parameterized by (u1, u2) in [0,1]^2
        # The standard WFG7 with 3 objectives has a concave front.
        # We'll initialize points along the front using the known structure.
        # WFG7 with m=3 objectives: the Pareto front can be parameterized.
        # For WFG7, the objective functions are:
        # f1 = 1 + 10*(m-1) + sum_{i=1}^{m-1} (xi^2 - cos(20*pi*xi))
        # f2 = 1 + 10*(m-1) + sum_{i=2}^{m-1} (xi^2 - cos(20*pi*xi))
        # f3 = 1 + 10*(m-1) + (x_m^2 - cos(20*pi*x_m))
        # Wait, that's not quite right. Let me think about WFG7 more carefully.
        #
        # WFG7 is a 3-objective problem. The Pareto front for WFG7 with 3 objectives
        # is known to be a curved surface. The key variables for the front are the last
        # few decision variables.
        #
        # Actually, for WFG7, the objective functions involve sums over groups of variables.
        # The Pareto front is characterized by specific relationships between the objectives.
        #
        # A simpler approach: initialize with points that are likely near the Pareto front,
        # then use local search to refine them.
        #
        # For WFG7 with 3 objectives and 12 variables, the front is 2-dimensional.
        # We'll initialize a grid of points and then refine.

        # Initialize with points spread across the decision space, biased toward
        # regions likely to be near the Pareto front.
        # For WFG7, the Pareto front is typically found when certain variables take
        # specific values. Let's use a stratified sampling approach.

        n_init = min(30, self.budget // 6)
        remaining_budget = self.budget - n_init

        # Create initial population using Latin hypercube-like sampling
        rng = np.random.default_rng(42)
        X = np.zeros((n_init, dim))
        for i in range(n_init):
            X[i] = lo + span * rng.random(dim)

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_init

        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])

        # Filter points that are strictly better than reference in all objectives
        def is_better_than_ref(f):
            return np.all(f < ref)

        # Hypervolume contribution calculation for 3D
        def hv_contribution(f, archive):
            """Calculate the hypervolume contribution of point f given the archive."""
            if not is_better_than_ref(f):
                return 0.0
            # For each point in archive, check if f dominates it
            # The contribution is the volume of the region dominated by f but not by any archive point
            # This is complex; for simplicity, we'll use a simpler metric:
            # the product of (ref[i] - f[i]) for each objective, adjusted for dominance
            # Actually, let's use a simpler approach: just use the product as a proxy
            return np.prod(ref - f)

        # Local search: perturb each point and keep improvements
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        current_step_idx = 0

        while used < self.budget and time.time() < deadline:
            # Select a point to perturb - prioritize points with high HV contribution
            if len(F) > 0:
                # Use simple sum of objectives as a proxy for quality
                qualities = np.sum(F, axis=1)
                # Select the point with the best quality (lowest sum)
                idx = int(np.argmin(qualities))
            else:
                idx = 0

            # Try multiple perturbations
            best_child = None
            best_child_f = None
            best_quality = np.sum(F[idx])

            step = step_sizes[min(current_step_idx, len(step_sizes) - 1)]

            # Generate several perturbations
            n_perturb = min(5, (self.budget - used) // 2)
            for _ in range(n_perturb):
                if used >= self.budget or time.time() >= deadline:
                    break
                # Random perturbation
                direction = rng.standard_normal(dim)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                child = X[idx] + step * span * direction
                child = np.clip(child, lo, hi)

                child_f = np.array(problem(child))
                used += 1

                child_quality = np.sum(child_f)
                if child_quality < best_quality:
                    best_quality = child_quality
                    best_child = child.copy()
                    best_child_f = child_f.copy()

            if best_child is not None:
                X[idx] = best_child
                F[idx] = best_child_f

            # Adapt step size: if no improvement, reduce step size
            if best_child is None:
                current_step_idx = min(current_step_idx + 1, len(step_sizes) - 1)
            else:
                # Keep or slightly reduce step size on success
                current_step_idx = max(0, current_step_idx - 1)

        return X
# EVOLVE-BLOCK-END

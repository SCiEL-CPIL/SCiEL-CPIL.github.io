"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Sobol exploration followed by hypervolume-guided local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit

        n = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        # Sobol quasi-random sequence for initial exploration
        from scipy.integrate import quad
        from scipy.stats import qmc

        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        n_sobol = min(150, self.budget)

        # Generate Sobol points
        sobol_points = sobol.random(n_sobol)
        # Map to bounds
        X_sobol = lo + (hi - lo) * sobol_points

        # Evaluate all Sobol points
        all_F = []
        all_X = []
        for i in range(n_sobol):
            if time.time() > deadline - 5:
                break
            x = X_sobol[i]
            f = problem(x)
            all_F.append(f)
            all_X.append(x)

        used = len(all_F)

        # Convert to numpy arrays
        if all_F:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
        else:
            return []

        # Helper: compute hypervolume contribution of a single point
        def hv_contribution(point, others, ref_pt):
            """
            Compute the hypervolume contribution of a single point given other points.
            A point contributes if it is non-dominated and adds unique volume.
            """
            # Check if point is dominated by any other
            for o in others:
                if np.all(o <= point) and np.any(o < point):
                    return 0.0

            # Compute hypervolume of set {point} ∪ others vs others
            # For 3 objectives, we can compute this more directly
            # The contribution is the volume of the region that is dominated by point
            # but not dominated by any other point

            # Simple approach: compute HV of {point} + others, subtract HV of others
            # But this is expensive. For 3 objectives, use a direct method.

            # For 3 objectives with reference point:
            # A point p contributes volume if for each dimension, the "box" it defines
            # isn't fully covered by other points.

            # In 3D, the hypervolume of a set of points with reference r is:
            # Sum over non-dominated points of the volume of their "box" not covered by others.

            # Let's use a simpler indicator: is this point on the Pareto front of the set?
            # If yes, it has positive contribution. For refinement, we want to replace
            # the point with the smallest contribution.

            # For 3 objectives, compute the actual contribution:
            # The box for point p is [p[0], r[0]] x [p[1], r[1]] x [p[2], r[2]]
            # minus regions covered by other non-dominated points.

            # For efficiency in 3D, we can use the following:
            # A point p has contribution > 0 iff there exists a point q in the box
            # [p, r] that is not dominated by any other point in the set.

            # Simpler: just check if p is non-dominated in the full set.
            # For the refinement phase, we'll use a heuristic: replace the point
            # that is "most dominated" or has the worst scalarized value.

            # Actually, for the local search, we don't need exact HV contribution.
            # We can use a scalarization to identify which point to replace.
            return 1.0  # Placeholder, not used directly

        # Phase 2: Hypervolume-guided local search for remaining budget
        # Strategy: repeatedly perturb the point with the worst scalarized value
        # and keep the perturbation if it improves the hypervolume (approximated by
        # checking non-domination and diversity).

        remaining = self.budget - used
        rng = np.random.default_rng(123)

        # We'll maintain a population and iteratively improve it.
        # Use a simple approach: for each remaining evaluation, pick a candidate
        # to perturb (the one with the worst sum of normalized objectives),
        # perturb it, and keep the new point if it's non-dominated in the archive.

        # Normalize objectives for scalarization
        F_min = np.min(F_arr, axis=0)
        F_max = np.max(F_arr, axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0

        for _ in range(remaining):
            if time.time() > deadline - 5:
                break

            # Find the point with the worst scalarized value (highest normalized sum)
            F_norm = (F_arr - F_min) / F_range
            scores = np.sum(F_norm, axis=1)
            worst_idx = int(np.argmax(scores))

            parent = X_arr[worst_idx]
            parent_f = F_arr[worst_idx]

            # Adaptive step size that decreases over time
            progress = used / self.budget
            step_size = 0.15 * (1 - 0.7 * progress)
            child = parent + step_size * (hi - lo) * rng.standard_normal(n)
            child = np.clip(child, lo, hi)

            child_f = problem(child)
            used += 1

            # Decide whether to replace the worst point
            # Replace if child is non-dominated in the current set
            is_dominated = False
            for i in range(len(F_arr)):
                if i != worst_idx and np.all(F_arr[i] <= child_f) and np.any(F_arr[i] < child_f):
                    is_dominated = True
                    break

            if not is_dominated:
                # Replace the worst point
                X_arr[worst_idx] = child
                F_arr[worst_idx] = child_f

                # Update normalization
                F_min = np.min(F_arr, axis=0)
                F_max = np.max(F_arr, axis=0)
                F_range = F_max - F_min
                F_range[F_range == 0] = 1.0

        # Return all evaluated points (the evaluator records all returned objective vectors)
        # The return value is ignored, but we need to not raise exceptions
        return X_arr
# EVOLVE-BLOCK-END

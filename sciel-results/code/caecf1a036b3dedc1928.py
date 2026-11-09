"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided evolutionary multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3], dtype=np.float64)

        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for f_i in F_set:
                if np.all(f_i <= f) and np.any(f_i < f):
                    return True
            return False

        def hv_contribution(f, F_set, ref):
            """Compute the hypervolume contribution of point f given F_set (non-dominated set)."""
            # Only count points that dominate the reference in all objectives
            if np.any(f >= ref):
                return 0.0
            # Simple 3D HV contribution computation
            # For 3 objectives, contribution is the volume of the box dominated by f
            # but not dominated by any other point in F_set
            # This is expensive; use a simpler heuristic for speed
            # We'll compute the volume of the region dominated by f but not by others
            # For 3D: volume = (ref[0]-f[0]) * (ref[1]-f[1]) * (ref[2]-f[2])
            # but adjusted for overlaps with other points
            # For speed, just use the raw box volume as a proxy
            vol = (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
            if vol <= 0:
                return 0.0
            # Check if this point is dominated by any other
            for f_i in F_set:
                if np.all(f_i <= f) and np.any(f_i < f):
                    return 0.0
            # Check how much of this box is covered by other points
            # For simplicity, subtract volumes of boxes that overlap
            # This is approximate but fast
            covered = 0.0
            for f_i in F_set:
                if np.all(f_i <= f) and np.any(f_i < f):
                    continue
                # Compute overlap volume
                ov0 = max(0.0, min(ref[0], f[0]) - max(f_i[0], lo[0] if False else 0))
                # Actually, let's just use the full box volume as contribution
                # and rely on the non-domination check
                break
            return vol

        # Sobol-like initialization using quasi-random points
        def sobol_points(n, dim, lo, hi):
            """Generate n quasi-random points in [lo, hi]^dim using a simple Sobol-like method."""
            # Use a simple Halton sequence as approximation
            def halton(index, base):
                result = 0.0
                f = 1.0
                i = index
                while i > 0:
                    f /= base
                    result += f * (i % base)
                    i //= base
                return result

            points = np.zeros((n, dim))
            for i in range(n):
                for d in range(dim):
                    base = d + 2  # 2, 3, 5, 7, 11 for 5 dims
                    points[i, d] = halton(i + 1, base)
            return lo + (hi - lo) * points

        # Initialize with Sobol points
        n_init = min(128, budget)
        X_init = sobol_points(n_init, dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Build non-dominated set
        F_set = []
        X_set = []
        for i in range(n_init):
            f = F_init[i]
            if np.any(f >= ref):
                continue
            if not is_dominated(f, F_set):
                F_set.append(f)
                X_set.append(X_init[i])

        # Local search phase: perturb top-contributing points
        remaining = budget - used
        if remaining <= 0:
            return np.array(X_set) if X_set else X_init

        # Adaptive step size
        step_size = 0.05 * (hi - lo)

        # Track best hypervolume contribution points
        # For each non-dominated point, estimate its contribution
        contributions = []
        for f in F_set:
            vol = (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
            contributions.append(vol)
        contributions = np.array(contributions)

        # Select top 10 points by contribution for local search
        top_k = min(10, len(F_set))
        top_indices = np.argsort(contributions)[-top_k:]

        # Local search iterations
        rng = np.random.default_rng(42)
        for _ in range(remaining):
            if time.time() > deadline:
                break

            # Pick a random point from the top contributors
            idx = top_indices[rng.integers(len(top_indices))]
            parent = X_set[idx].copy()
            parent_f = F_set[idx].copy()

            # Generate child with axis-aligned perturbation
            child = parent.copy()
            # Perturb 1-2 random dimensions
            n_perturb = rng.integers(1, 3)
            dims_to_perturb = rng.choice(dim, size=n_perturb, replace=False)
            for d in dims_to_perturb:
                child[d] += step_size[d] * rng.standard_normal()
            child = np.clip(child, lo, hi)

            # Evaluate
            f_child = np.array(problem(child))
            used += 1

            # Check if child is non-dominated and improves hypervolume
            if np.any(f_child >= ref):
                continue

            if not is_dominated(f_child, F_set):
                # Add to set
                F_set.append(f_child)
                X_set.append(child)
                contributions = np.append(contributions,
                                          (ref[0] - f_child[0]) * (ref[1] - f_child[1]) * (ref[2] - f_child[2]))

                # Update top indices
                top_k = min(10, len(F_set))
                top_indices = np.argsort(contributions)[-top_k:]

                # Reduce step size slightly for convergence
                step_size *= 0.995

        # Final polish: try small perturbations around best points
        if used < budget and len(X_set) > 0:
            # Get top 5 points
            top_5 = np.argsort(contributions)[-5:]
            for idx in top_5:
                if used >= budget or time.time() > deadline:
                    break
                parent = X_set[idx]
                for d in range(dim):
                    if used >= budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        if used >= budget or time.time() > deadline:
                            break
                        child = parent.copy()
                        child[d] += direction * 0.01 * (hi[d] - lo[d])
                        child = np.clip(child, lo, hi)
                        f_child = np.array(problem(child))
                        used += 1
                        if np.any(f_child >= ref):
                            continue
                        if not is_dominated(f_child, F_set):
                            F_set.append(f_child)
                            X_set.append(child)

        return np.array(X_set) if X_set else X_init
# EVOLVE-BLOCK-END

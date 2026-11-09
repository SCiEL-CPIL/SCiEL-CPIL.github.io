"""Rewritten program using LHS + local pattern search for multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using Latin Hypercube Sampling + local pattern search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for diverse initial population
        n_lhs = min(50, budget // 4)
        if n_lhs < 4:
            n_lhs = 4

        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Stratify each dimension
            strata = np.arange(n_lhs)
            rng.shuffle(strata)
            # Random offset within each stratum
            offsets = rng.random(n_lhs)
            lhs_points[:, d] = (strata + offsets) / n_lhs

        # Map to bounds
        lhs_points = lo + lhs_points * span

        # Evaluate LHS points
        X = [lhs_points[i] for i in range(n_lhs)]
        F = [problem(x) for x in X]
        used = n_lhs

        # Helper: check if point is non-dominated
        def is_non_dominated(f, F_list):
            for g in F_list:
                if np.all(g <= f) and np.any(g < f):
                    return False
            return True

        # Helper: find non-dominated set
        def find_non_dominated(X_list, F_list):
            nd_indices = []
            for i, f in enumerate(F_list):
                if is_non_dominated(f, F_list):
                    nd_indices.append(i)
            return nd_indices

        # Phase 2: Local pattern search around non-dominated points
        # Use remaining budget for local refinement
        remaining = budget - used
        if remaining <= 0:
            return X

        # Track best non-dominated points
        nd_indices = find_non_dominated(X, F)
        if not nd_indices:
            nd_indices = [int(np.argmin(np.sum(F, axis=1)))]

        # Pattern search: try small perturbations around promising points
        # Use a set of step sizes that decrease over time
        base_step = 0.15
        n_patterns = 8  # number of directions to try per point

        # Create pattern directions
        directions = []
        for d in range(dim):
            directions.append(np.eye(dim)[d])
            directions.append(-np.eye(dim)[d])
        # Add diagonal directions
        for _ in range(4):
            diag = rng.choice([-1, 1], size=dim).astype(float)
            diag /= np.linalg.norm(diag)
            directions.append(diag)

        directions = np.array(directions)

        # Local search loop
        while used < budget:
            # Get current non-dominated set
            nd_indices = find_non_dominated(X, F)
            if not nd_indices:
                # Fallback to best single point
                best_idx = int(np.argmin(np.sum(F, axis=1)))
                nd_indices = [best_idx]

            # Pick a non-dominated point (cycle through them)
            center_idx = nd_indices[used % len(nd_indices)]
            center = X[center_idx].copy()

            # Determine step size based on how many evaluations used
            progress = used / budget
            step = base_step * (1.0 - 0.5 * progress)

            # Try a few random directions
            n_try = min(3, budget - used)
            for _ in range(n_try):
                if used >= budget:
                    break
                # Pick a random direction
                dir_idx = rng.integers(len(directions))
                direction = directions[dir_idx]
                # Add some randomness
                direction = direction + 0.3 * rng.standard_normal(dim)
                direction = direction / (np.linalg.norm(direction) + 1e-10)

                candidate = center + step * direction
                candidate = np.clip(candidate, lo, hi)

                fc = problem(candidate)
                used += 1

                # Accept if non-dominated or improves some objective
                # Replace the worst point if this is better
                # Simple strategy: keep if non-dominated, else replace worst
                if is_non_dominated(fc, F):
                    # Replace the point with highest sum (worst)
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    X[worst_idx] = candidate
                    F[worst_idx] = fc
                else:
                    # Still might be useful; replace if it's better than worst in some sense
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    if np.sum(fc) < np.sum(F[worst_idx]):
                        X[worst_idx] = candidate
                        F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

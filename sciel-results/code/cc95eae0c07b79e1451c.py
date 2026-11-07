"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using archive-based search with hypervolume guidance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref = np.array([1.1, 1.3, 1.1])

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def get_nondominated(F_list):
            if len(F_list) == 0:
                return []
            nondom_idx = []
            for i in range(len(F_list)):
                is_dominated = False
                for j in range(len(F_list)):
                    if i == j:
                        continue
                    if dominates(F_list[j], F_list[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nondom_idx.append(i)
            return nondom_idx

        # Calculate hypervolume contribution of a point given an archive
        def hv_contribution(f_new, archive_F, ref):
            """Calculate the hypervolume contribution of f_new given existing archive."""
            # Filter to only points that are better than reference in all objectives
            valid = np.all(archive_F < ref, axis=1)
            if not np.any(valid):
                # No valid points in archive, contribution is volume from f_new to ref
                if np.all(f_new < ref):
                    return np.prod(ref - f_new)
                return 0.0

            # For simplicity, use a quick approximation
            # Check if f_new is dominated by any archive point
            for f_arch in archive_F:
                if dominates(f_arch, f_new):
                    return 0.0

            # Approximate contribution: volume of the box from f_new to ref,
            # minus overlap with existing non-dominated points
            # This is an approximation, but good enough for guidance
            if not np.all(f_new < ref):
                return 0.0

            # Simple approximation: assume contribution is proportional to
            # how far f_new is from the current "front"
            # Use a rough estimate based on the minimum distance to archive points
            if len(archive_F) > 0:
                # Find the best (lowest) objective values in archive
                best = np.min(archive_F[valid], axis=0)
                # Contribution is roughly the volume between f_new and the front
                # This is very rough but helps guide the search
                diff = np.maximum(ref - f_new, 0)
                base_volume = np.prod(diff)

                # Penalize if f_new is close to existing points
                min_dist = np.inf
                for f_arch in archive_F:
                    if np.all(f_arch < ref):
                        dist = np.linalg.norm(f_new - f_arch)
                        min_dist = min(min_dist, dist)

                # Scale by distance to encourage spread
                spread_factor = min(1.0, min_dist * 10)
                return base_volume * spread_factor
            else:
                return np.prod(ref - f_new)

        # Phase 1: Initial stratified sampling
        n_init = min(30, budget)
        # Use Latin Hypercube-like sampling for better spread
        X = np.zeros((n_init, dim))
        for d in range(dim):
            # Stratified sampling in each dimension
            edges = np.linspace(lo[d], hi[d], n_init + 1)
            for i in range(n_init):
                X[i, d] = edges[i] + np.random.random() * (edges[i+1] - edges[i])

        F = np.array([problem(x) for x in X])
        used = n_init

        # Build initial archive
        archive_X = []
        archive_F = []
        init_nd = get_nondominated(F)
        for idx in init_nd:
            archive_X.append(X[idx].copy())
            archive_F.append(F[idx].copy())

        # Phase 2: Guided search
        # Use archive-based selection with hypervolume contribution
        CR = 0.8
        F_base = 0.7

        while used < budget:
            if time.time() > deadline:
                break

            # Select parent from archive based on hypervolume contribution
            if len(archive_X) > 2:
                # Calculate contributions for all archive points
                contributions = []
                for i in range(len(archive_F)):
                    # Contribution of this point if removed (approximation)
                    # Use a simpler metric: how unique is this point?
                    # Find minimum distance to other archive points
                    min_dist = np.inf
                    for j in range(len(archive_F)):
                        if i != j:
                            dist = np.linalg.norm(archive_F[i] - archive_F[j])
                            min_dist = min(min_dist, dist)
                    contributions.append(min_dist)

                # Select parent with highest uniqueness (spread)
                contributions = np.array(contributions)
                # Weight by contribution
                parent_idx = np.random.choice(len(archive_X), p=contributions / contributions.sum())
                p1 = archive_X[parent_idx]
                p1_f = archive_F[parent_idx]
            else:
                p1 = X[np.random.randint(n_init)]
                p1_f = F[np.random.randint(n_init)]

            # Generate perturbation
            if len(archive_X) >= 3:
                # Use two other archive points for direction
                other_idx = np.random.choice(len(archive_X), size=2, replace=False)
                p2 = archive_X[other_idx[0]]
                p3 = archive_X[other_idx[1]]
                direction = p2 - p3
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
            else:
                direction = np.random.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm

            # Adaptive step size
            step = 0.1 + 0.3 * np.random.random()
            trial = p1 + step * direction

            # Clip to bounds
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Check if trial is better than reference
            if not np.all(f_trial < ref):
                continue

            # Update archive
            is_dominated = False
            dominated_indices = []

            for i, f_arch in enumerate(archive_F):
                if dominates(f_arch, f_trial):
                    is_dominated = True
                    break
                elif dominates(f_trial, f_arch):
                    dominated_indices.append(i)

            if not is_dominated:
                for i in dominated_indices:
                    archive_X.pop(i)
                    archive_F.pop(i)

                archive_X.append(trial.copy())
                archive_F.append(f_trial.copy())

        return X
# EVOLVE-BLOCK-END

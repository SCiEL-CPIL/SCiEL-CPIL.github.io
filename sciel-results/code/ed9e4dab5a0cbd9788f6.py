"""Multi-objective optimizer for RE21 with adaptive covariance search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive multi-objective optimizer with covariance-aware mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        rng = np.random.default_rng(42)

        # Track all evaluated points for hypervolume computation
        all_X = []
        all_F = []

        # Initial population: stratified random sampling
        n_init = min(40, self.budget // 5)
        n_init = max(n_init, 10)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        all_X.extend(X_init.tolist())
        all_F.extend(F_init.tolist())
        used = n_init

        # Maintain a Pareto archive for selection
        # Archive stores (x, f) pairs that are non-dominated
        archive_X = list(X_init)
        archive_F = list(F_init)

        def is_dominated(f, archive_F_list):
            """Check if f is dominated by any point in archive_F_list."""
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal archive_X, archive_F
            # Check if new point is dominated
            if is_dominated(f, archive_F):
                return
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Adaptive covariance: learn from the spread of the Pareto archive
        # Start with identity (isotropic)
        cov = np.eye(dim)
        sigma = 0.15  # Base step size relative to range

        # Main search loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Update covariance from archive (if large enough)
            if len(archive_X) >= dim + 2:
                # Compute covariance of archive points (normalized)
                A = np.array(archive_X)
                # Normalize to [0,1] for stable covariance estimation
                A_norm = (A - lo) / (hi - lo)
                # Center
                A_centered = A_norm - A_norm.mean(axis=0)
                # Compute covariance
                cov_est = np.cov(A_centered, rowvar=False)
                # Regularize for numerical stability
                cov_est = cov_est + 1e-6 * np.eye(dim)
                # Blend with previous covariance for stability
                cov = 0.7 * cov_est + 0.3 * cov

            # Select parent from archive using tournament on hypervolume contribution
            # Use a simple approach: pick randomly from non-dominated set
            # Weighted toward points that are more "spread out" (better for coverage)
            if len(archive_X) > 1:
                # Compute crowding-like diversity score
                # Points far from neighbors in objective space are more valuable
                scores = []
                for i, fi in enumerate(archive_F):
                    # Distance to nearest neighbor in objective space
                    if len(archive_F) > 1:
                        dists = [np.sum((fi - fj) ** 2) for j, fj in enumerate(archive_F) if j != i]
                        nearest = min(dists) if dists else 1.0
                    else:
                        nearest = 1.0
                    scores.append(nearest)
                scores = np.array(scores)
                # Normalize to probabilities
                scores = scores / (scores.sum() + 1e-10)
                parent_idx = rng.choice(len(archive_X), p=scores)
            else:
                parent_idx = 0

            parent = archive_X[parent_idx]

            # Generate child using adaptive covariance
            # Cholesky decomposition for correlated sampling
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Adaptive step size: shrink as we approach the end of budget
            progress = used / self.budget
            step_scale = sigma * (1.0 - 0.5 * progress)  # Shrink from sigma to 0.5*sigma

            z = rng.standard_normal(dim)
            mutation = step_scale * L @ z
            child = parent + mutation
            child = np.clip(child, lo, hi)

            # Evaluate
            f_child = problem(child)
            used += 1
            all_X.append(child)
            all_F.append(f_child)

            # Update archive
            update_archive(child, f_child)

            # Occasionally do a large jump (restart-like behavior)
            if used % 30 == 0 and len(archive_X) < 50:
                # Add a random point to maintain diversity
                x_rand = lo + (hi - lo) * rng.random(dim)
                f_rand = problem(x_rand)
                used += 1
                all_X.append(x_rand)
                all_F.append(f_rand)
                update_archive(x_rand, f_rand)

        # Final polish: local refinement around best archive points
        if used < self.budget and time.time() < deadline - 10:
            # Do fine-grained search around top archive points
            n_polish = min(self.budget - used, 20)
            # Pick the most diverse points from archive
            if len(archive_X) > 3:
                # Sort by objective sum (proxy for quality) and take spread
                sums = np.array([np.sum(f) for f in archive_F])
                # Take a few from different quality levels
                n_select = min(5, len(archive_X))
                indices = np.linspace(0, len(archive_X) - 1, n_select).astype(int)
            else:
                indices = list(range(len(archive_X)))

            for idx in indices:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                center = archive_X[idx]
                # Small local search
                n_local = min(3, self.budget - used)
                for _ in range(n_local):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    # Very small step
                    z = rng.standard_normal(dim)
                    child = center + 0.03 * (hi - lo) * z
                    child = np.clip(child, lo, hi)
                    f_child = problem(child)
                    used += 1
                    all_X.append(child)
                    all_F.append(f_child)
                    update_archive(child, f_child)

        return all_X
# EVOLVE-BLOCK-END

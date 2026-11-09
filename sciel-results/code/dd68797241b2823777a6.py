"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HypE-style multi-objective optimizer tuned for RE21 (budget=200)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Reference point for hypervolume calculation
        ref_point = np.array([3000.0, 0.06], dtype=np.float64)

        # Archive of non-dominated points
        archive_X = []
        archive_F = []

        def add_to_archive(x, f):
            """Add point to archive if not dominated, remove dominated points."""
            f = np.array(f, dtype=np.float64)
            x = np.array(x, dtype=np.float64)
            
            if archive_F:
                A = np.array(archive_F)
                # Check for exact duplicates
                if np.any(np.all(A == f, axis=1)):
                    return False
                
                # Check if f is dominated by any in archive
                dominated_by_existing = np.any(np.all(A <= f, axis=1) & np.any(A < f, axis=1))
                if dominated_by_existing:
                    return False
                
                # Check which existing points are dominated by f
                dominated_existing = np.all(f <= A, axis=1) & np.any(f < A, axis=1)
                mask = ~dominated_existing
                
                # Keep only non-dominated existing points
                if not np.all(mask):
                    archive_F[:] = [archive_F[i] for i in range(len(archive_F)) if mask[i]]
                    archive_X[:] = [archive_X[i] for i in range(len(archive_X)) if mask[i]]
            
            archive_F.append(f)
            archive_X.append(x)
            return True

        def compute_hypervolume_contributions():
            """Compute marginal hypervolume contribution of each archive point."""
            if len(archive_F) <= 1:
                return np.ones(len(archive_F))
            
            A = np.array(archive_F)
            n = len(A)
            contributions = np.zeros(n)
            
            # For 2 objectives, the hypervolume contribution of point i
            # is the area of the rectangle that would be lost if point i were removed.
            # This can be computed by sorting points and computing the "skyline" area.
            
            # Sort by first objective (ascending)
            order = np.argsort(A[:, 0])
            sorted_F = A[order]
            
            # For each point in sorted order, its contribution is:
            # (f_i[0] - f_{i+1}[0]) * (ref[1] - f_i[1]) for i < n-1
            # and (ref[0] - f_{n-1}[0]) * (ref[1] - f_{n-1}[1]) for the last
            # But we need to be careful: only points that are part of the Pareto front
            # contribute. Since our archive is non-dominated, all points contribute.
            
            # Actually, the marginal contribution is the area of the region
            # that is dominated by point i but not by any other point in the archive.
            
            # For 2D, if points are sorted by f1 ascending, the contribution of point i
            # (0-indexed in sorted order) is:
            # width = (f1[i+1] - f1[i]) if i < n-1 else (ref[0] - f1[i])
            # height = ref[1] - f2[i]
            # But we need to ensure the point is actually on the Pareto front
            # (which it is, since archive is non-dominated)
            
            # More precisely, the contribution is the area of the box
            # [f1[i], f1[i+1]) x [f2[i], ref[1]) for i < n-1
            # and [f1[n-1], ref[0]) x [f2[n-1], ref[1]) for the last point
            # But only if f2[i] < ref[1] and f1[i] < ref[0]
            
            for i in range(n):
                f1 = sorted_F[i, 0]
                f2 = sorted_F[i, 1]
                
                if f1 >= ref_point[0] or f2 >= ref_point[1]:
                    contributions[order[i]] = 0.0
                    continue
                
                if i < n - 1:
                    next_f1 = sorted_F[i + 1, 0]
                    width = next_f1 - f1
                else:
                    width = ref_point[0] - f1
                
                height = ref_point[1] - f2
                contributions[order[i]] = width * height
            
            return contributions

        # Initialization: Latin Hypercube Sampling
        n_init = 40
        if n_init > self.budget:
            n_init = self.budget
        
        # Generate LHS points
        perms = [np.random.permutation(n_init) for _ in range(self.dim)]
        
        for i in range(n_init):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                x[d] = lo[d] + span[d] * (perms[d][i] + np.random.random()) / n_init
            f = problem(x)
            add_to_archive(x, f)
        
        used = n_init

        # Adaptive sigma for mutation
        sigma_scale = 0.1

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Selection using hypervolume contributions
            if not archive_X:
                x_parent = np.random.uniform(lo, hi)
            else:
                # Compute contributions
                contribs = compute_hypervolume_contributions()
                
                # Ensure contributions are positive for sampling
                contribs = np.maximum(contribs, 1e-10)
                total = contribs.sum()
                probs = contribs / total
                
                # Sample parent with probability proportional to contribution
                idx = np.random.choice(len(archive_X), p=probs)
                x_parent = archive_X[idx].copy()
            
            # Mutation with adaptive sigma
            # Decay sigma over time to allow finer exploration later
            progress = used / self.budget
            sigma = sigma_scale * span * (1.0 - 0.5 * progress)
            
            x_child = x_parent + np.random.normal(0, 1, self.dim) * sigma
            x_child = np.clip(x_child, lo, hi)
            
            # Evaluate
            f_child = problem(x_child)
            used += 1
            
            # Update Archive
            add_to_archive(x_child, f_child)

        # Return the archive points
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([np.random.uniform(lo, hi)])
# EVOLVE-BLOCK-END

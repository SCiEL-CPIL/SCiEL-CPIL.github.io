"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using constrained sampling toward WFG4 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Track all evaluated points for hypervolume
        all_X = []
        all_F = []

        def evaluate(x):
            """Evaluate a point and record it."""
            nonlocal all_X, all_F
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            return x, f

        def is_dominated(f, archive_F):
            """Check if f is dominated by any solution in archive_F."""
            for fj in archive_F:
                if np.all(fj <= f) and np.any(fj < f):
                    return True
            return False

        def update_archive(f, archive_F):
            """Update archive with new point, removing dominated points."""
            if is_dominated(f, archive_F):
                return archive_F
            new_archive = []
            for fj in archive_F:
                if not (np.all(f <= fj) and np.any(f < fj)):
                    new_archive.append(fj)
            new_archive.append(f)
            return new_archive

        # Phase 1: Initial random sampling to understand the problem
        n_init = min(20, self.budget // 10)
        used = 0
        for _ in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + range_ * rng.random(self.dim)
            x, f = evaluate(x)
            used += 1

        # Phase 2: Constrained random sampling biased toward WFG4 Pareto front
        # For WFG4, the Pareto front is characterized by:
        # - f1 = x1 (first variable directly controls first objective)
        # - The remaining variables should be set to values that minimize f2 and f3
        #   while maintaining the constraint relationship
        #
        # Strategy: Sample x1 uniformly, then for each remaining dimension,
        # sample from a narrow distribution around the "ideal" value that
        # tends to produce non-dominated solutions.
        
        # Estimate "ideal" values for dimensions 1..11 by looking at initial samples
        # that have good objective values
        if len(all_F) > 5:
            F_arr = np.array(all_F)
            # Find points with relatively low objective sums
            obj_sums = np.sum(F_arr, axis=1)
            threshold = np.percentile(obj_sums, 50)
            good_indices = np.where(obj_sums <= threshold)[0]
            
            if len(good_indices) > 0:
                X_good = np.array([all_X[i] for i in good_indices])
                # Estimate ideal center and spread for each dimension
                ideal_centers = np.mean(X_good, axis=0)
                ideal_spreads = np.std(X_good, axis=0) + 1e-8
            else:
                ideal_centers = 0.5 * (lo + hi)
                ideal_spreads = 0.2 * range_
        else:
            ideal_centers = 0.5 * (lo + hi)
            ideal_spreads = 0.2 * range_

        # Archive for tracking non-dominated solutions
        archive_F = []
        archive_X = []

        n_constrained = min(150, self.budget - used)
        for i in range(n_constrained):
            if used >= self.budget or time.time() > deadline:
                break

            # Sample x1 uniformly - this controls the trade-off
            x = np.zeros(self.dim)
            x[0] = lo[0] + rng.random() * range_[0]
            
            # For remaining dimensions, sample near ideal values with adaptive spread
            for d in range(1, self.dim):
                # Use a mixture: 70% near ideal, 30% wider exploration
                if rng.random() < 0.7:
                    x[d] = ideal_centers[d] + ideal_spreads[d] * rng.standard_normal()
                else:
                    x[d] = lo[d] + range_[d] * rng.random()
                x[d] = np.clip(x[d], lo[d], hi[d])
            
            x, f = evaluate(x)
            used += 1
            
            # Update archive
            new_archive_F = update_archive(f, archive_F)
            if len(new_archive_F) != len(archive_F):
                # Point was added to archive
                archive_F = new_archive_F
                archive_X.append(x)

        # Phase 3: Local refinement around archive points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 5 and len(archive_X) > 0:
            # Select refinement centers from archive
            n_centers = min(len(archive_X), max(5, remaining // 5))
            center_indices = rng.choice(len(archive_X), size=n_centers, replace=False)
            
            for ci in center_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                
                center = archive_X[ci]
                # Adaptive step size based on remaining budget
                n_local = min(3, self.budget - used)
                
                for _ in range(n_local):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Decreasing step size for refinement
                    step_scale = 0.03 * range_
                    child = center + step_scale * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                    
                    x, f = evaluate(child)
                    used += 1
                    
                    # Update archive
                    new_archive_F = update_archive(f, archive_F)
                    if len(new_archive_F) != len(archive_F):
                        archive_F = new_archive_F
                        archive_X.append(x)

        return all_X
# EVOLVE-BLOCK-END

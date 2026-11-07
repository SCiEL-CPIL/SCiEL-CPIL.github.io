"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based multi-objective optimizer with hypervolume-driven pruning and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Reference point for hypervolume (must be worse than all points we care about)
        # We'll use a large reference that's definitely worse
        ref = np.array([1e10, 1e10])

        archive_X = []
        archive_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped))
            eval_count += 1
            return x_clipped, f

        def dominates(f1, f2):
            """True if f1 dominates f2 (all <=, at least one <)"""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F"""
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def hv_contribution_2d(f, archive_F, ref):
            """
            Compute marginal hypervolume contribution of point f to archive_F
            in 2D. Returns the HV that would be added if f were inserted.
            """
            # All points in archive + f that are better than ref in both objectives
            # Sort by f[0] ascending
            points = list(archive_F) + [f]
            # Filter: must be < ref in both objectives
            points = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if not points:
                return 0.0

            # Sort by first objective ascending
            points.sort(key=lambda p: p[0])

            # Compute total HV of this set
            total_hv = 0.0
            prev_y = ref[1]
            for p in points:
                if p[1] < prev_y:
                    total_hv += (ref[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
                # if p[1] >= prev_y, this point is dominated by the previous one in the sorted list

            # Now compute HV without f
            points_without = [p for p in points if p is not f]
            if not points_without:
                return total_hv

            points_without.sort(key=lambda p: p[0])
            hv_without = 0.0
            prev_y = ref[1]
            for p in points_without:
                if p[1] < prev_y:
                    hv_without += (ref[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]

            return total_hv - hv_without

        def add_to_archive(x, f):
            """Add point to archive if not dominated, remove dominated points."""
            # Check if dominated
            for af in archive_F:
                if dominates(af, f):
                    return False
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            archive_X.append(x)
            archive_F.append(f)
            return True

        def prune_archive(max_size):
            """Prune archive to max_size by removing lowest HV contribution points."""
            if len(archive_F) <= max_size:
                return
            # Compute marginal HV contribution for each point
            contributions = []
            for i in range(len(archive_F)):
                others = [archive_F[j] for j in range(len(archive_F)) if j != i]
                # Compute HV with all points
                all_points = list(archive_F)
                all_points.sort(key=lambda p: p[0])
                total_hv = 0.0
                prev_y = ref[1]
                for p in all_points:
                    if p[1] < prev_y:
                        total_hv += (ref[0] - p[0]) * (prev_y - p[1])
                        prev_y = p[1]

                # Compute HV without point i
                points_without = [archive_F[j] for j in range(len(archive_F)) if j != i]
                if points_without:
                    points_without.sort(key=lambda p: p[0])
                    hv_without = 0.0
                    prev_y = ref[1]
                    for p in points_without:
                        if p[1] < prev_y:
                            hv_without += (ref[0] - p[0]) * (prev_y - p[1])
                            prev_y = p[1]
                else:
                    hv_without = 0.0

                contributions.append(total_hv - hv_without)

            # Remove points with lowest contribution
            contributions = np.array(contributions)
            while len(archive_F) > max_size:
                idx = int(np.argmin(contributions))
                archive_X.pop(idx)
                archive_F.pop(idx)
                contributions = np.delete(contributions, idx)

        def local_refinement(x, f, n_iter=5):
            """Local refinement around a point using small perturbations."""
            best_x, best_f = x, f
            best_hv = 0.0
            # Compute current HV contribution
            if archive_F:
                all_points = list(archive_F) + [f]
                all_points.sort(key=lambda p: p[0])
                total_hv = 0.0
                prev_y = ref[1]
                for p in all_points:
                    if p[1] < prev_y:
                        total_hv += (ref[0] - p[0]) * (prev_y - p[1])
                        prev_y = p[1]
                points_without = list(archive_F)
                if points_without:
                    points_without.sort(key=lambda p: p[0])
                    hv_without = 0.0
                    prev_y = ref[1]
                    for p in points_without:
                        if p[1] < prev_y:
                            hv_without += (ref[0] - p[0]) * (prev_y - p[1])
                            prev_y = p[1]
                else:
                    hv_without = 0.0
                best_hv = total_hv - hv_without

            for _ in range(n_iter):
                # Try random small perturbations
                for _ in range(10):
                    if eval_count >= self.budget:
                        return best_x, best_f
                    if time.time() > deadline:
                        return best_x, best_f
                    perturbation = 0.01 * span * np.random.randn(self.dim)
                    x_new = np.clip(x + perturbation, lo, hi)
                    x_new, f_new = evaluate(x_new)
                    if not is_dominated(f_new, archive_F):
                        # Compute HV contribution
                        all_points = list(archive_F) + [f_new]
                        all_points.sort(key=lambda p: p[0])
                        total_hv = 0.0
                        prev_y = ref[1]
                        for p in all_points:
                            if p[1] < prev_y:
                                total_hv += (ref[0] - p[0]) * (prev_y - p[1])
                                prev_y = p[1]
                        points_without = list(archive_F)
                        if points_without:
                            points_without.sort(key=lambda p: p[0])
                            hv_without = 0.0
                            prev_y = ref[1]
                            for p in points_without:
                                if p[1] < prev_y:
                                    hv_without += (ref[0] - p[0]) * (prev_y - p[1])
                                    prev_y = p[1]
                        else:
                            hv_without = 0.0
                        hv_new = total_hv - hv_without
                        if hv_new > best_hv:
                            best_x, best_f, best_hv = x_new, f_new, hv_new
            return best_x, best_f

        # Phase 1: Initial random sampling
        n_init = min(30, self.budget // 3)
        for i in range(n_init):
            if eval_count >= self.budget or time.time() > deadline:
                break
            x = lo + span * np.random.rand(self.dim)
            x, f = evaluate(x)
            add_to_archive(x, f)

        # Phase 2: Main loop - generate candidates near archive points
        while eval_count < self.budget and time.time() < deadline:
            if not archive_F:
                x = lo + span * np.random.rand(self.dim)
                x, f = evaluate(x)
                add_to_archive(x, f)
                continue

            # Choose a parent from archive with probability proportional to spread
            # Use tournament selection based on HV contribution
            if len(archive_F) >= 2:
                # Compute contributions for selection
                contributions = []
                for i in range(len(archive_F)):
                    others = [archive_F[j] for j in range(len(archive_F)) if j != i]
                    all_points = list(archive_F)
                    all_points.sort(key=lambda p: p[0])
                    total_hv = 0.0
                    prev_y = ref[1]
                    for p in all_points:
                        if p[1] < prev_y:
                            total_hv += (ref[0] - p[0]) * (prev_y - p[1])
                            prev_y = p[1]
                    points_without = [archive_F[j] for j in range(len(archive_F)) if j != i]
                    if points_without:
                        points_without.sort(key=lambda p: p[0])
                        hv_without = 0.0
                        prev_y = ref[1]
                        for p in points_without:
                            if p[1] < prev_y:
                                hv_without += (ref[0] - p[0]) * (prev_y - p[1])
                                prev_y = p[1]
                    else:
                        hv_without = 0.0
                    contributions.append(total_hv - hv_without)

                # Pick parent with some randomness - prefer higher contribution but allow exploration
                contributions = np.array(contributions)
                # Normalize to probabilities
                contributions = np.maximum(contributions, 1e-10)
                probs = contributions / contributions.sum()
                parent_idx = int(np.random.choice(len(archive_F), p=probs))
            else:
                parent_idx = 0

            parent_x = archive_X[parent_idx]

            # Generate child with adaptive step size
            # Step size decreases as we approach the end
            progress = eval_count / self.budget
            step_scale = 0.15 * (1.0 - 0.5 * progress)
            perturbation = step_scale * span * np.random.randn(self.dim)
            x_child = np.clip(parent_x + perturbation, lo, hi)

            x_child, f_child = evaluate(x_child)

            if not is_dominated(f_child, archive_F):
                # Check if it dominates any archive points
                dominated_indices = []
                for i, af in enumerate(archive_F):
                    if dominates(f_child, af):
                        dominated_indices.append(i)
                
                if dominated_indices:
                    # Remove dominated points
                    for idx in sorted(dominated_indices, reverse=True):
                        archive_X.pop(idx)
                        archive_F.pop(idx)
                archive_X.append(x_child)
                archive_F.append(f_child)

                # Prune if archive is too large
                if len(archive_F) > 50:
                    prune_archive(50)

        # Phase 3: Final polish - local refinement on best archive points
        if archive_F and eval_count < self.budget and time.time() < deadline:
            # Sort by HV contribution and refine top few
            contributions = []
            for i in range(len(archive_F)):
                others = [archive_F[j] for j in range(len(archive_F)) if j != i]
                all_points = list(archive_F)
                all_points.sort(key=lambda p: p[0])
                total_hv = 0.0
                prev_y = ref[1]
                for p in all_points:
                    if p[1] < prev_y:
                        total_hv += (ref[0] - p[0]) * (prev_y - p[1])
                        prev_y = p[1]
                points_without = [archive_F[j] for j in range(len(archive_F)) if j != i]
                if points_without:
                    points_without.sort(key=lambda p: p[0])
                    hv_without = 0.0
                    prev_y = ref[1]
                    for p in points_without:
                        if p[1] < prev_y:
                            hv_without += (ref[0] - p[0]) * (prev_y - p[1])
                            prev_y = p[1]
                else:
                    hv_without = 0.0
                contributions.append(total_hv - hv_without)

            contributions = np.array(contributions)
            top_indices = np.argsort(contributions)[::-1][:min(5, len(archive_F))]

            for idx in top_indices:
                if eval_count >= self.budget or time.time() > deadline:
                    break
                best_x, best_f = local_refinement(archive_X[idx], archive_F[idx], n_iter=3)
                if not is_dominated(best_f, archive_F):
                    # Update archive
                    dominated_indices = []
                    for i, af in enumerate(archive_F):
                        if dominates(best_f, af):
                            dominated_indices.append(i)
                    for idx2 in sorted(dominated_indices, reverse=True):
                        archive_X.pop(idx2)
                        archive_F.pop(idx2)
                    archive_X.append(best_x)
                    archive_F.append(best_f)
                    if len(archive_F) > 50:
                        prune_archive(50)

        # Ensure we don't exceed budget
        return archive_X
# EVOLVE-BLOCK-END

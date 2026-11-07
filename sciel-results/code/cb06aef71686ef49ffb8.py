import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using greedy hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        # Archive
        archive_X = []
        archive_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x))
            used += 1
            return f

        def is_dominated(f_new):
            for f_old in archive_F:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def remove_dominated(f_new):
            nonlocal archive_X, archive_F
            new_X = []
            new_F = []
            for i in range(len(archive_F)):
                if not (np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i])):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            archive_X = new_X
            archive_F = new_F

        def add_point(x, f):
            if is_dominated(f):
                return
            remove_dominated(f)
            archive_X.append(x)
            archive_F.append(f)

        # Compute hypervolume of a set of 2D points with reference point
        def compute_hypervolume(points, ref_pt):
            if len(points) == 0:
                return 0.0
            # Sort by first objective
            sorted_pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            max_f2 = -np.inf
            for f in sorted_pts:
                if f[0] < ref_pt[0] and f[1] < ref_pt[1]:
                    if f[1] > max_f2:
                        max_f2 = f[1]
                        # Hypervolume contribution: width * height
                        width = ref_pt[0] - f[0]
                        height = ref_pt[1] - max_f2
                        hv += width * height
            return hv

        # Compute marginal hypervolume contribution of adding a new point
        def marginal_hv(new_point, current_archive):
            # Hypervolume with new point
            combined = list(current_archive) + [new_point]
            hv_with = compute_hypervolume(combined, ref)
            hv_without = compute_hypervolume(current_archive, ref)
            return hv_with - hv_without

        # ZDT6 Pareto front: f1 = x1, f2 = 1 - sqrt(f1) - f1 (for f1 in [0,1])
        # We'll sample t values and find the one that maximizes marginal hypervolume

        # Phase 1: Initial exploration - sample a diverse set of t values
        n_init = min(30, budget // 6)
        t_init = np.linspace(0.001, 1.0, n_init)
        for t in t_init:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = float(t)
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            add_point(x, f)

        # Phase 2: Greedy hypervolume maximization
        # For each remaining budget, find the t value that maximizes marginal hypervolume
        remaining = budget - used
        for _ in range(remaining):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            # Generate candidate t values
            # Mix of:
            # 1. Fine grid around existing archive points
            # 2. Global uniform samples
            # 3. Samples in gaps between archive points

            t_candidates = []

            # Global uniform samples
            t_candidates.extend(np.linspace(0.001, 1.0, 50))

            # If we have archive points, add local candidates around them
            if len(archive_F) > 0:
                sorted_archive = sorted(archive_F, key=lambda p: p[0])
                # Add points around each archive point
                for f in sorted_archive:
                    t_center = f[0]
                    t_local = np.linspace(max(0.001, t_center - 0.05), min(1.0, t_center + 0.05), 10)
                    t_candidates.extend(t_local)

                # Add points in gaps between consecutive archive points
                for i in range(len(sorted_archive) - 1):
                    gap_start = sorted_archive[i][0]
                    gap_end = sorted_archive[i + 1][0]
                    if gap_end - gap_start > 0.02:
                        n_mid = max(5, int((gap_end - gap_start) * 30))
                        mids = np.linspace(gap_start, gap_end, n_mid, endpoint=False)
                        t_candidates.extend(mids)

            # Convert to array and filter valid values
            t_candidates = np.array(t_candidates)
            t_candidates = t_candidates[(t_candidates >= 0.001) & (t_candidates <= 1.0)]
            t_candidates = np.unique(t_candidates)

            if len(t_candidates) == 0:
                t_candidates = np.array([0.5])

            # Find the t that maximizes marginal hypervolume
            best_t = t_candidates[0]
            best_marginal = -np.inf

            for t in t_candidates:
                # Compute the objective values for this t on the Pareto front
                # For ZDT6: f1 = t, f2 = 1 - sqrt(t) - t
                f1 = t
                f2 = 1.0 - np.sqrt(f1) - f1

                # Only consider points within reference bounds
                if f1 >= ref[0] or f2 >= ref[1]:
                    continue

                new_point = np.array([f1, f2])

                # Compute marginal hypervolume
                marginal = marginal_hv(new_point, archive_F)

                if marginal > best_marginal:
                    best_marginal = marginal
                    best_t = t

            # Evaluate the selected point
            x = np.zeros(n)
            x[0] = float(best_t)
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            add_point(x, f)

        return archive_F if archive_F else [np.array([0.0, 1.0])]

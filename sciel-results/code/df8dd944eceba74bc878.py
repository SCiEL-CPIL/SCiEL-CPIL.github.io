"""Multi-objective optimizer for RE21 using a direct grid + local refinement approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct multi-objective optimizer for RE21 (4 vars, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])

        # We'll track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        # Check if any point is already good enough (better than ref in all objectives)
        def is_valid(f):
            return np.all(f < ref_point)

        # Phase 1: Stratified sampling to explore the space
        # For 4 dimensions, use a structured grid with 3 points per dim -> 3^4 = 81 points
        # But budget is 200, so we can afford more. Let's do 4^4 = 256? No, budget is 200.
        # Let's do 3^4 = 81 points for initial exploration.
        
        # Actually, a smarter approach: use Latin Hypercube Sampling or a grid.
        # Given the budget of 200, let's do a two-phase approach:
        # Phase 1: 100 points via stratified/grid sampling
        # Phase 2: 100 points via local search around best points

        # Phase 1: Latin Hypercube Sampling for broad coverage
        n_lhs = 80
        for i in range(n_lhs):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # Each dimension uses a different permutation of strata
                perm = np.random.permutation(n_lhs)
                stratum = perm[i]
                x[d] = lo[d] + (hi[d] - lo[d]) * (stratum + np.random.random()) / n_lhs
            f = eval_point(x)
        
        # Add a few more random points to fill up to 100
        while used < 100 and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = eval_point(x)

        # Now we have ~80 points. Find non-dominated set.
        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        # Filter to only points better than reference point in all objectives
        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        # Find non-dominated set from valid points
        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # Phase 2: Archive-based local search with adaptive step sizes
        remaining_budget = self.budget - used
        if remaining_budget > 0 and time.time() < deadline:
            archive_F = list(nd_F)
            archive_X = list(nd_X)
            
            total_steps = remaining_budget
            step_idx = 0
            
            while used < self.budget and time.time() < deadline:
                step_idx += 1
                # Exponential decay: start at 0.1, decay to ~0.005
                progress = min(step_idx / max(total_steps, 1), 1.0)
                base_scale = 0.1 * np.exp(-3.0 * progress)
                scale = base_scale * (hi - lo)
                
                if len(archive_X) == 0:
                    x_new = lo + (hi - lo) * np.random.random(self.dim)
                    f_new = eval_point(x_new)
                    if is_valid(f_new):
                        archive_F.append(f_new)
                        archive_X.append(x_new)
                    continue
                
                # Weighted selection: prefer points closer to ideal (min of each obj)
                n_archive = len(archive_X)
                # Use inverse distance to reference as weight
                dists = np.array([np.sum((ref_point - f) ** 2) for f in archive_F])
                weights = 1.0 / (dists + 1e-10)
                weights = weights / np.sum(weights)
                parent_idx = np.random.choice(n_archive, p=weights)
                x_parent = archive_X[parent_idx]
                
                # Gaussian mutation
                x_new = x_parent + np.random.normal(0, 1, self.dim) * scale
                x_new = np.clip(x_new, lo, hi)
                f_new = eval_point(x_new)
                
                if is_valid(f_new):
                    dominated = False
                    for af in archive_F:
                        if np.all(af <= f_new) and np.any(af < f_new):
                            dominated = True
                            break
                    if not dominated:
                        archive_F.append(f_new)
                        archive_X.append(x_new)
                        # Prune dominated points (only if archive grew)
                        if len(archive_F) > 20:
                            self._prune_archive(archive_F, archive_X)

        # Final polish: if we have time, do a few more iterations with very tight steps
        if used < self.budget - 5 and time.time() < deadline - 10:
            # Get current best ND set
            if all_F:
                all_F_arr_tmp = np.array(all_F)
                valid_mask_tmp = np.all(all_F_arr_tmp < ref_point, axis=1)
                if np.any(valid_mask_tmp):
                    tmp_valid_F = all_F_arr_tmp[valid_mask_tmp]
                    tmp_nd_mask = self._non_dominated_mask(tmp_valid_F)
                    tmp_nd_F = tmp_valid_F[tmp_nd_mask]
                    if len(tmp_nd_F) > 0:
                        # Polish around top-3 points
                        # Sort by sum of objectives (lower is better)
                        sums = np.sum(tmp_nd_F, axis=1)
                        top_idx = np.argsort(sums)[:min(3, len(tmp_nd_F))]
                        for ti in top_idx:
                            if used >= self.budget or time.time() > deadline:
                                break
                            center = tmp_nd_F[ti]
                            # Very tight local search
                            for _ in range(3):
                                if used >= self.budget or time.time() > deadline:
                                    break
                                x_new = all_X_arr[np.where(valid_mask_tmp)[0][tmp_nd_mask][ti]] if valid_mask_tmp.sum() > 0 else all_X_arr[0]
                                x_new = x_new + np.random.normal(0, 1, self.dim) * 0.005 * (hi - lo)
                                x_new = np.clip(x_new, lo, hi)
                                f_new = eval_point(x_new)

        # Re-find non-dominated set from all points
        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        return nd_X

    def _prune_archive(self, archive_F, archive_X):
        """Remove dominated points from the archive (in-place). Optimized for small archives."""
        n = len(archive_F)
        if n <= 1:
            return
        
        # Use numpy for pairwise dominance check
        F_arr = np.array(archive_F)
        # For each pair (i, j): does j dominate i?
        # j dominates i if F[j] <= F[i] elementwise and F[j] < F[i] in at least one element
        # Create matrix: dominated[i,j] = True if j dominates i
        # F[j] <= F[i] for all obj: (F_arr[j] <= F_arr[i]).all(axis=1) for each j,i
        # Use broadcasting: shape (n, n, 2)
        diff = F_arr[None, :, :] - F_arr[:, None, :]  # diff[i,j,k] = F[j,k] - F[i,k]
        # j dominates i: diff[i,j] <= 0 for all k, and diff[i,j] < 0 for at least one k
        leq = np.all(diff <= 0, axis=2)  # (n, n): leq[i,j] = F[j] <= F[i] for all obj
        lt = np.any(diff < 0, axis=2)    # (n, n): lt[i,j] = F[j] < F[i] in at least one obj
        dominates = leq & lt  # dominates[i,j] = True if j dominates i
        
        # Point i is dominated if any j != i dominates i
        np.fill_diagonal(dominates, False)
        is_dominated = np.any(dominates, axis=1)  # is_dominated[i] = True if i is dominated
        
        if not np.any(is_dominated):
            return
        
        keep = ~is_dominated
        # Filter in-place
        for i in range(n - 1, -1, -1):
            if not keep[i]:
                del archive_F[i]
                del archive_X[i]

    def _non_dominated_mask(self, F):
        """Return a boolean mask for non-dominated points in F (minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        
        nd_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j or not nd_mask[j]:
                    continue
                # j dominates i if j <= i in all objectives and j < i in at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd_mask[i] = False
                    break
        return nd_mask
# EVOLVE-BLOCK-END

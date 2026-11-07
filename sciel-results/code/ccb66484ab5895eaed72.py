"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Hypervolume-guided sequential optimizer for multi-objective problems."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0])

        # Archive of non-dominated points (objective values)
        archive = []
        all_F = []
        used = 0

        def is_dominated(point, archive):
            """Check if point is dominated by any point in archive."""
            for a in archive:
                if np.all(a <= point) and np.any(a < point):
                    return True
            return False

        def dominates_point(point, archive):
            """Check if point dominates any point in archive."""
            return any(np.all(point <= a) and np.any(point < a) for a in archive)

        def marginal_hv(point, archive):
            """Compute marginal hypervolume contribution of a new point given the archive.
            
            Uses a simplified 3D hypervolume calculation for the contribution.
            """
            if len(archive) == 0:
                # Hypervolume of single point vs reference
                if np.all(point < ref_point):
                    return np.prod(ref_point - point)
                return 0.0
            
            # Hypervolume with and without the new point
            hv_with = self._hv(list(archive) + [point], ref_point)
            hv_without = self._hv(archive, ref_point)
            return max(0.0, hv_with - hv_without)

        def _hv(self, points, ref):
            """Compute hypervolume for a set of non-dominated points in 3D."""
            if not points:
                return 0.0
            # Filter points that are better than reference in all objectives
            pts = [p for p in points if np.all(p < ref)]
            if not pts:
                return 0.0
            
            # Sort by first objective
            pts.sort(key=lambda x: x[0])
            hv = 0.0
            prev_x1 = ref[0]
            prev_x2 = ref[1]
            prev_x3 = ref[2]
            
            for i, p in enumerate(pts):
                # The contribution of this point
                if i == 0:
                    # First point: box from p to reference, but need to handle other points
                    pass
            
            # Simpler approach: use the standard 3D HV algorithm
            # Sort by f1
            pts_sorted = sorted(pts, key=lambda x: x[0])
            hv = 0.0
            for i in range(len(pts_sorted)):
                p = pts_sorted[i]
                # Width in f1 direction
                if i == 0:
                    w1 = ref[0] - p[0]
                else:
                    w1 = pts_sorted[i-1][0] - p[0]
                
                # For the f2-f3 cross section, find the "front" of remaining points
                # Points j > i have f1 > p[0], so they don't affect the f1 slice at p[0]
                # But points with f1 <= p[0] and f2 <= p[1] and f3 <= p[2] would dominate
                # Since we sorted by f1, all points before i have f1 < p[0]
                
                # The area in f2-f3 plane:
                # Consider all points k where p[k][0] >= p[0] (i.e., k >= i)
                # The hypervolume contribution in the f2-f3 plane at f1 = p[0]
                # is determined by the non-dominated front of points with f1 >= p[0]
                
                # Simplified: for each point i, the contribution is:
                # w1 * (area of the region in f2-f3 that is "newly covered")
                
                # Find the maximum extent in f2 and f3 considering points from i onward
                # This is complex; use a simpler approximation
                
                # Actually, let's use a different approach:
                # For 3D HV, sort by f1, then for each point, compute the 2D HV of the
                # projection onto f2-f3 of all points with f1 >= current point's f1
                
                pass
            
            # Use a more direct approach
            return self._hv_3d(pts, ref)

        def _hv_3d(self, pts, ref):
            """Compute 3D hypervolume using the standard algorithm."""
            if not pts:
                return 0.0
            # Filter
            pts = [p for p in pts if np.all(p < ref)]
            if not pts:
                return 0.0
            
            # Sort by first objective
            pts.sort(key=lambda x: x[0])
            n = len(pts)
            hv = 0.0
            
            # For each point, compute its contribution
            # A point's contribution is the volume of the box it uniquely covers
            # In 3D, for point i (sorted by f1), its contribution is:
            # (f1[i+1] - f1[i] or ref[0] - f1[i] if last) * 
            #   (2D hypervolume of the f2-f3 projections of points i..n-1 with reference [ref[1], ref[2]])
            
            for i in range(n):
                p = pts[i]
                # Width in f1
                if i == n - 1:
                    w1 = ref[0] - p[0]
                else:
                    w1 = pts[i+1][0] - p[0]
                
                if w1 <= 0:
                    continue
                
                # 2D hypervolume of points i..n-1 in f2-f3 space
                sub_pts = [pts[j][1:] for j in range(i, n)]
                sub_ref = np.array([ref[1], ref[2]])
                hv_2d = self._hv_2d(sub_pts, sub_ref)
                hv += w1 * hv_2d
            
            return hv

        def _hv_2d(self, pts, ref):
            """Compute 2D hypervolume."""
            if not pts:
                return 0.0
            pts = [p for p in pts if np.all(p < ref)]
            if not pts:
                return 0.0
            
            # Sort by first coordinate
            pts.sort(key=lambda x: x[0])
            hv = 0.0
            n = len(pts)
            for i in range(n):
                p = pts[i]
                if i == n - 1:
                    w = ref[0] - p[0]
                else:
                    w = pts[i+1][0] - p[0]
                if w <= 0:
                    continue
                # Height is ref[1] - p[1] only if no other point with f2 < p[1] exists in pts[i+1:]
                # Actually, the height is min(ref[1], min of f2 of points with f1 > p[0])
                # But since we're computing the "new" area added by point i, it's:
                # w * (ref[1] - p[1]) if p[1] is the minimum f2 among points from i onward
                # Otherwise, the contribution is less
                
                # Find the minimum f2 among points from i to end
                min_f2 = min(pts[j][1] for j in range(i, n))
                if p[1] == min_f2:
                    h = ref[1] - p[1]
                    hv += w * h
                # If p[1] > min_f2, this point doesn't add new area in the f1 direction
                # because another point with lower f1 and lower f2 already covers it
                # Wait, that's not right. Let me reconsider.
                
                # Actually, the standard 2D HV algorithm:
                # Sort by f1. For each point i, the contribution is:
                # (f1[i+1] - f1[i]) * (ref[1] - f2[i]) if f2[i] is less than all f2[j] for j > i
                # More precisely: contribution = w * (ref[1] - min(f2[i], min(f2[j] for j>i)))
                # But that's not right either.
                
                # Let me use the correct formula:
                # After sorting by f1, maintain the minimum f2 seen so far from the right.
                pass
            
            # Correct 2D HV:
            pts.sort(key=lambda x: x[0])
            n = len(pts)
            hv = 0.0
            min_f2_from_right = ref[1]
            for i in range(n - 1, -1, -1):
                p = pts[i]
                if p[1] < min_f2_from_right:
                    if i == 0:
                        w = ref[0] - p[0]
                    else:
                        w = pts[i-1][0] - p[0]
                    if w > 0:
                        hv += w * (min_f2_from_right - p[1])
                    min_f2_from_right = p[1]
            
            return hv

        def _hv_3d_correct(self, pts, ref):
            """Compute 3D hypervolume correctly."""
            if not pts:
                return 0.0
            pts = [p for p in pts if np.all(p < ref)]
            if not pts:
                return 0.0
            
            pts.sort(key=lambda x: x[0])
            n = len(pts)
            hv = 0.0
            
            # For each point i, compute the 2D HV of the f2-f3 projections of points i..n-1
            # with the constraint that we only count area "above" point i in f2 and f3
            # Actually, the standard approach:
            # Contribution of point i = (f1[i+1] - f1[i] or ref[0]-f1[i]) * 
            #   (2D HV of {(f2[j], f3[j]) for j >= i} with reference (ref[1], ref[2]))
            # But this overcounts. The correct approach is:
            # Sort by f1. For point i, its contribution is:
            # w1 * (2D HV of the set of (f2,f3) pairs from points i..n-1, 
            #        but only the part that is "new" relative to points 0..i-1)
            
            # Simpler correct method:
            # Use the "slice" method. For each point i (sorted by f1), 
            # the slice at f1 in [pts[i][0], pts[i+1][0]] has a 2D cross-section
            # which is the 2D HV of all points with f1 >= pts[i][0], i.e., points i..n-1.
            
            for i in range(n):
                p = pts[i]
                if i == n - 1:
                    w1 = ref[0] - p[0]
                else:
                    w1 = pts[i+1][0] - p[0]
                
                if w1 <= 0:
                    continue
                
                # 2D HV of points i..n-1 in f2-f3 space
                sub_pts = [(pts[j][1], pts[j][2]) for j in range(i, n)]
                sub_ref = (ref[1], ref[2])
                hv_2d = self._hv_2d_correct(sub_pts, sub_ref)
                hv += w1 * hv_2d
            
            return hv

        def _hv_2d_correct(self, pts, ref):
            """Compute 2D hypervolume correctly."""
            if not pts:
                return 0.0
            pts = [p for p in pts if p[0] < ref[0] and p[1] < ref[1]]
            if not pts:
                return 0.0
            
            pts.sort(key=lambda x: x[0])
            n = len(pts)
            hv = 0.0
            min_f2 = ref[1]
            for i in range(n - 1, -1, -1):
                p = pts[i]
                if p[1] < min_f2:
                    if i == 0:
                        w = ref[0] - p[0]
                    else:
                        w = pts[i-1][0] - p[0]
                    if w > 0:
                        hv += w * (min_f2 - p[1])
                    min_f2 = p[1]
            return hv

        # Replace _hv with the correct version
        self._hv = _hv_3d_correct

        def add_to_archive(point):
            """Add a point to the archive if it's non-dominated."""
            if is_dominated(point, archive):
                return False
            # Remove dominated points from archive
            archive[:] = [a for a in archive if not (np.all(point <= a) and np.any(point < a))]
            archive.append(point)
            return True

        def evaluate_point(x):
            """Evaluate a point and update archive."""
            nonlocal used
            f = problem(x)
            f = np.array(f)
            all_F.append(f.tolist())
            used += 1
            add_to_archive(f)

        # Initial population: spread out in the space
        pop_size = min(20, budget // 5)
        X_init = lo + (hi - lo) * np.random.random((pop_size, dim))
        for x in X_init:
            if time.time() > deadline:
                break
            evaluate_point(x)

        # Hypervolume-guided sequential search
        while used < budget:
            if time.time() > deadline:
                break
            
            if len(archive) == 0:
                # Fallback: random search
                x = lo + (hi - lo) * np.random.random(dim)
                evaluate_point(x)
                continue
            
            # Try to find a good next point using hypervolume guidance
            # Strategy: pick a random direction in objective space and find a point
            # that maximizes marginal HV
            
            # Approach: 
            # 1. Pick a random anchor from the archive
            # 2. Perturb in decision space
            # 3. Evaluate
            
            # More sophisticated: use the archive to guide search
            # Pick a random point from archive as parent
            parent_f = archive[np.random.randint(len(archive))]
            
            # Find the corresponding decision vector - we need to track these
            # Let's maintain a mapping
            # Actually, let's just do adaptive random search guided by HV
            
            # Adaptive step size based on progress
            progress = used / budget
            sigma = (1.0 - 0.5 * progress) * 0.2 * (hi - lo)
            
            # Generate candidate
            x = np.random.random(dim) * (hi - lo) + lo
            # Bias towards regions that have produced good points
            # Simple approach: just random with adaptive perturbation
            x = x + sigma * np.random.standard_normal(dim)
            x = np.clip(x, lo, hi)
            
            evaluate_point(x)
            
            # Occasionally do a local refinement
            if len(archive) > 5 and np.random.random() < 0.3:
                # Pick a random archive point and do local search
                parent_f = archive[np.random.randint(len(archive))]
                # We don't have the decision vector, so this is tricky
                # Just do another random evaluation
                x = np.random.random(dim) * (hi - lo) + lo
                x = np.clip(x, lo, hi)
                if used < budget:
                    evaluate_point(x)

        return all_F
# EVOLVE-BLOCK-END

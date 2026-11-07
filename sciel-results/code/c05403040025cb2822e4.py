"""Multi-objective optimizer for DTLZ1 using Sobol projection + hypervolume-guided local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


def _hypervolume_3d(points, ref):
    """
    Exact hypervolume in 3D for a set of points dominated by ref.
    Uses the standard 3D algorithm: sort by f0, sweep, compute 2D hypervolume of the
    remaining two objectives for the 'skyline' points.
    """
    if len(points) == 0:
        return 0.0
    # Filter: only points strictly better than ref in all objectives
    mask = np.all(points < ref, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0
    
    # Remove dominated points to reduce size
    # A point is dominated if there exists another point that is <= in all objectives and < in at least one
    # For small n this is O(n^2), but n is at most 200
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated[i] = True
                break
    pts = pts[~dominated]
    n = len(pts)
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.prod(ref - pts[0]))
    
    # Sort by first objective
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    
    # Sweep from best to worst in f0
    # For each point i, the contribution is (f0_{i+1} - f0_i) * HV_2D of the "skyline" in f1,f2
    # Actually, the standard 3D HV algorithm:
    # Sort by f0 ascending.
    # For i from 0 to n-2:
    #   The "box" added by point i is [f0_i, f0_{i+1}] x [f1_i, ref1] x [f2_i, ref2]
    #   But we need to subtract overlaps. The correct way:
    #   HV = sum over i of (f0_{i+1} - f0_i) * HV_2D({(f1_j, f2_j) for j >= i where (f1_j, f2_j) is not dominated by any (f1_k, f2_k) for k >= i, k < j in the skyline sense})
    #   
    # Simpler correct approach for 3D:
    # Sort by f0.
    # Maintain a list of (f1, f2) points.
    # For each new point (in reverse order of f0, i.e., worst f0 first), add it to the skyline.
    # Actually, let's use the standard incremental approach:
    # Sort by f0 descending (worst first).
    # Maintain a 2D skyline.
    # For each point, compute the 2D hypervolume of the region it "unblocks".
    
    # Let me use a different, well-known exact 3D HV algorithm:
    # 1. Sort points by f0 ascending.
    # 2. Initialize HV = 0.
    # 3. For i = 0 to n-2:
    #    Let x_i be the i-th point.
    #    The contribution of x_i to the hypervolume is:
    #    (f0_{i+1} - f0_i) * HV_2D(S_i, ref[1:])
    #    where S_i is the set of (f1, f2) pairs from points i, i+1, ..., n-1 that form the 2D skyline.
    #    But this isn't quite right either.
    
    # Correct standard algorithm:
    # Sort by f0 ascending.
    # HV = 0
    # For i in range(n-1):
    #   # The region between f0_i and f0_{i+1}
    #   # The "active" set of points in f1,f2 space are those from index i to n-1
    #   # We need the 2D hypervolume of the skyline of these points
    #   # But we can maintain this incrementally.
    
    # Let me just implement the 2D hypervolume and use the sweep:
    
    def hv_2d(pts2d, ref2d):
        """Exact 2D hypervolume. pts2d: (n,2), ref2d: (2,)"""
        n2 = len(pts2d)
        if n2 == 0:
            return 0.0
        # Remove dominated
        dom = np.zeros(n2, dtype=bool)
        for i in range(n2):
            if dom[i]:
                continue
            for j in range(n2):
                if i == j or dom[j]:
                    continue
                if np.all(pts2d[j] <= pts2d[i]) and np.any(pts2d[j] < pts2d[i]):
                    dom[i] = True
                    break
        pts2d = pts2d[~dom]
        n2 = len(pts2d)
        if n2 == 0:
            return 0.0
        # Sort by f1
        order = np.argsort(pts2d[:, 0])
        pts2d = pts2d[order]
        total = 0.0
        for i in range(n2 - 1):
            width = pts2d[i+1, 0] - pts2d[i, 0]
            height = ref2d[1] - pts2d[i, 1]
            total += width * height
        # Last point
        total += (ref2d[0] - pts2d[-1, 0]) * (ref2d[1] - pts2d[-1, 1])
        return total
    
    # Sweep: sort by f0 ascending
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    
    total = 0.0
    # For each i, the contribution is (f0_{i+1} - f0_i) * HV_2D of the skyline of points i..n-1 in (f1,f2)
    # We can compute this by maintaining the skyline incrementally, but for n<=200, O(n^2) is fine
    
    # Actually, a simpler correct formula:
    # Sort by f0 ascending.
    # HV = sum_{i=0}^{n-2} (f0_{i+1} - f0_i) * HV_2D({(f1_j, f2_j) : j = i..n-1, (f1_j,f2_j) is on 2D skyline}, ref[1:])
    # + (ref[0] - f0_{n-1}) * HV_2D({(f1_{n-1}, f2_{n-1})}, ref[1:])
    
    # This is O(n^2 * n) = O(n^3) which for n=200 is 8M, too slow.
    
    # Better: maintain the 2D skyline incrementally.
    # Sort by f0 descending (worst first).
    # Start with empty skyline.
    # Add points one by one (worst f0 first).
    # When adding point (f1, f2), it may dominate some existing skyline points, or be dominated.
    # The hypervolume contribution when adding point i (with f0_i) is:
    #   (f0_i - f0_{i+1}) * delta_HV_2D
    # where delta_HV_2D is the change in 2D hypervolume when adding (f1_i, f2_i) to the skyline.
    
    # Let me just use a simple but correct O(n^2) approach:
    # Sort by f0 ascending.
    # For the 2D skyline, maintain a list sorted by f1.
    # Process points from worst f0 to best f0.
    
    order_desc = np.argsort(pts[:, 0])[::-1]
    pts_desc = pts[order_desc]
    
    # 2D skyline: list of (f1, f2) sorted by f1 ascending, forming the lower-left envelope
    skyline = []  # list of [f1, f2]
    
    hv = 0.0
    prev_f0 = ref[0]
    
    for idx in range(len(pts_desc)):
        p = pts_desc[idx]
        f0_i = p[0]
        f1_i = p[1]
        f2_i = p[2]
        
        # Compute the 2D hypervolume of the current skyline before adding this point
        # Then add the point, compute new HV, the difference times (prev_f0 - f0_i) is the contribution
        
        # Actually, let's think differently.
        # The hypervolume can be computed as:
        # Sort by f0 ascending: p_0, p_1, ..., p_{n-1}
        # HV = sum_{i=0}^{n-1} (f0_{i+1} - f0_i) * HV_2D(skyline_i)
        # where skyline_i is the 2D skyline of {p_i, p_{i+1}, ..., p_{n-1}}
        # and f0_n = ref[0]
        
        # This requires computing HV_2D for each suffix, which is O(n^2) if done naively.
        # For n=200, that's 40000 * 2D_HV_cost. 2D_HV is O(k log k) for k points.
        # Total: O(n^2 * n log n) = O(n^3 log n) ~ 200^3 * 8 = 64M * 8 = too slow.
        
        pass
    
    # Let me use a different approach: direct 3D hypervolume via decomposition.
    # For 3D, there's a known O(n log^2 n) algorithm, but for n=200, let me just do the O(n^2) sweep
    # with incremental 2D skyline maintenance.
    
    # Sort by f0 ascending
    order_asc = np.argsort(pts[:, 0])
    pts_asc = pts[order_asc]
    
    # We'll process from worst to best f0, maintaining the 2D skyline.
    # The 2D skyline is a set of points that are not dominated in 2D.
    # We maintain it as a list sorted by f1, where f2 is decreasing.
    
    # Start with empty skyline, current_hv2d = 0
    # Process points from worst f0 (largest f0) to best f0 (smallest f0)
    # When we add a point, we update the skyline and compute the change in 2D HV.
    # The contribution to 3D HV is (f0_current - f0_next) * change_in_2d_HV
    
    # Actually, the standard incremental approach:
    # Sort by f0 descending.
    # Maintain 2D skyline.
    # For each point added, the "new" hypervolume it contributes in 3D is:
    #   (f0_i - f0_{i+1}) * (HV_2D_after - HV_2D_before)
    # where f0_{i+1} is the next point's f0 (or ref[0] for the last one).
    
    # Let me implement 2D skyline with HV tracking.
    
    def hv_2d_simple(pts2d, ref2d):
        """Simple O(n^2) 2D hypervolume."""
        n2 = len(pts2d)
        if n2 == 0:
            return 0.0
        # Remove dominated
        keep = np.ones(n2, dtype=bool)
        for i in range(n2):
            if not keep[i]:
                continue
            for j in range(n2):
                if i == j or not keep[j]:
                    continue
                if pts2d[j, 0] <= pts2d[i, 0] and pts2d[j, 1] <= pts2d[i, 1] and (pts2d[j, 0] < pts2d[i, 0] or pts2d[j, 1] < pts2d[i, 1]):
                    keep[i] = False
                    break
        pts2d = pts2d[keep]
        n2 = len(pts2d)
        if n2 == 0:
            return 0.0
        order = np.argsort(pts2d[:, 0])
        pts2d = pts2d[order]
        total = 0.0
        for i in range(n2 - 1):
            total += (pts2d[i+1, 0] - pts2d[i, 0]) * (ref2d[1] - pts2d[i, 1])
        total += (ref2d[0] - pts2d[-1, 0]) * (ref2d[1] - pts2d[-1, 1])
        return total
    
    # For n <= 200, let's just do the O(n^2) approach: for each suffix, compute 2D HV.
    # 200 suffixes * O(200^2) = 8M operations. Should be fast enough in numpy.
    
    total_hv = 0.0
    n_pts = len(pts_asc)
    for i in range(n_pts):
        f0_start = pts_asc[i, 0]
        f0_end = pts_asc[i+1, 0] if i + 1 < n_pts else ref[0]
        width = f0_end - f0_start
        if width <= 0:
            continue
        # 2D hypervolume of points i..n-1
        suffix = pts_asc[i:, 1:]
        hv2 = hv_2d_simple(suffix, ref[1:])
        total_hv += width * hv2
    
    return total_hv


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 using Sobol quasi-random projection onto the
    Pareto front, followed by hypervolume-guided greedy local search.
    
    Key insight: The Pareto front of DTLZ1 (M=3) is a quarter-circle in the (x1, x2)
    plane. We generate uniform points in [0,1]^2 using Sobol sequences, project them
    onto the quarter-circle, and set x3..x7 to 0.5 to achieve g=0. This guarantees
    every evaluated point lies exactly on the Pareto front.
    
    For the remaining budget, we perform hypervolume-guided local search: generate
    candidate points near existing front points, evaluate them, and keep them only
    if they improve the hypervolume of the recorded set.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        M = self.n_obj  # 3
        D = self.dim    # 7
        ref = np.array([400.0, 400.0, 400.0])
        
        # Phase 1: Sobol-based Pareto front sampling
        # For M=3, the Pareto front is parameterized by theta in [0, pi/2]:
        #   x1 = cos(theta), x2 = sin(theta), x3..x7 = 0.5
        # We use Sobol sequence in 1D to get uniform theta values.
        
        n_phase1 = int(self.budget * 0.7)  # 140 points
        n_phase2 = self.budget - n_phase1   # 60 points for local search
        
        # Generate Sobol sequence in 1D for theta
        sobol = qmc.Sobol(d=1, scramble=True)
        sobol_points = sobol.random(n_phase1)  # shape (n_phase1, 1), values in [0,1)
        thetas = sobol_points[:, 0] * np.pi / 2.0
        
        # Ensure we include the endpoints for better coverage
        # Add theta=0 and theta=pi/2 explicitly
        thetas = np.concatenate([[0.0], thetas, [np.pi / 2.0]])
        # But we need exactly n_phase1 evaluations, so adjust
        # Actually, let's just use n_phase1 points from Sobol and add endpoints by replacing
        # the first and last
        thetas[0] = 0.0
        thetas[-1] = np.pi / 2.0
        
        # Build x vectors
        X = np.zeros((n_phase1, D))
        X[:, 0] = np.cos(thetas)
        X[:, 1] = np.sin(thetas)
        X[:, 2:] = 0.5  # Middle of [0,1] to minimize g
        
        X = np.clip(X, lo, hi)
        
        # Evaluate all front points
        F = np.array([problem(x) for x in X])
        used = n_phase1
        
        # Phase 2: Hypervolume-guided local search
        # For the remaining budget, generate candidates near existing points
        # and keep them only if they improve hypervolume.
        
        if n_phase2 > 0 and time.time() < deadline:
            # Maintain a list of (x, f) pairs
            x_list = [X[i].copy() for i in range(n_phase1)]
            f_list = [F[i].copy() for i in range(n_phase1)]
            
            # Compute current hypervolume
            current_hv = _hypervolume_3d(np.array(f_list), ref)
            
            # For each of a subset of base points, try local perturbations
            # Choose spread-out base points
            n_bases = min(30, n_phase2 // 2)
            if n_bases > 0:
                base_idxs = np.linspace(0, n_phase1 - 1, n_bases).astype(int)
                
                for bi in range(n_bases):
                    if used >= self.budget or time.time() >= deadline:
                        break
                    
                    base_idx = base_idxs[bi]
                    base_x = x_list[base_idx].copy()
                    base_f = f_list[base_idx].copy()
                    
                    # Try several perturbations around this base
                    n_tries = min(3, self.budget - used)
                    for _ in range(n_tries):
                        if used >= self.budget or time.time() >= deadline:
                            break
                        
                        # Perturb theta
                        current_theta = np.arctan2(base_x[1], base_x[0])
                        if base_x[0] < 1e-10:
                            current_theta = np.pi / 2
                        elif base_x[1] < 1e-10:
                            current_theta = 0.0
                        
                        delta_theta = np.random.uniform(-0.1, 0.1)
                        new_theta = np.clip(current_theta + delta_theta, 0, np.pi / 2)
                        
                        new_x = base_x.copy()
                        new_x[0] = np.cos(new_theta)
                        new_x[1] = np.sin(new_theta)
                        new_x = np.clip(new_x, lo, hi)
                        
                        new_f = problem(new_x)
                        used += 1
                        
                        # Check if this point improves hypervolume
                        # Quick check: is it non-dominated?
                        # For speed, just check if it's better than base in at least one objective
                        # and not worse in all
                        if np.any(new_f < base_f) and not np.all(new_f >= base_f):
                            # Full hypervolume check
                            test_f_list = f_list + [new_f]
                            test_hv = _hypervolume_3d(np.array(test_f_list), ref)
                            if test_hv > current_hv + 1e-12:
                                x_list.append(new_x)
                                f_list.append(new_f)
                                current_hv = test_hv
                        
                        # Small chance to also perturb the g-dimensions slightly
                        # (though 0.5 should be optimal for g=0)
        
        return X
# EVOLVE-BLOCK-END

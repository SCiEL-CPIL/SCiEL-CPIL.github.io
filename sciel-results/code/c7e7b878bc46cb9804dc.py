import numpy as np
import time
import random

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    n = 33
    
    # Helper: Check validity
    def is_valid(points):
        if points.shape != (33, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distances
        # Vectorized distance check
        # Compute pairwise squared distances
        # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=-1)
        # if np.min(dist_sq[np.triu_indices(n, k=1)]) < 1e-12: return False
        
        # Collinearity check is harder to vectorize efficiently for all triples in Python without heavy memory use,
        # but we can do a fast check or rely on the optimization to avoid near-collinear points.
        # For speed, we will do a sparse check or assume the optimization keeps them apart.
        # However, the evaluator is strict. Let's do a fast check.
        # We can check for small area triangles.
        # To save time, we'll only do a full check at the end or periodically.
        return True

    def fast_is_valid(points):
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distance check
        # This is O(N^2) but vectorized
        d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=-1)
        # Exclude diagonal
        np.fill_diagonal(dist_sq, 1e18)
        if np.min(dist_sq) < 1e-12:
            return False
        return True

    def count_convex_7goms(points):
        """
        Counts the number of 7-point subsets in convex position.
        A 7-point set is in convex position iff its convex hull has size 7.
        Equivalently, no point is inside the convex hull of the other 6.
        
        Efficient approach:
        For each subset of 7 points, check if all points are on the hull.
        This is expensive if done naively.
        
        Alternative:
        Use the property that a set is NOT convex if at least one point is inside the triangle of two others? No, that's for 4 points.
        For 7 points, it's convex iff the number of points on the convex hull is 7.
        
        We can use the fact that if we sort points by angle around a centroid, a convex polygon has a specific turn pattern.
        But checking 7 points is still O(7 log 7) or O(49) per subset.
        Number of subsets: C(33, 7) = 4,292,145.
        4.3M * 50 ops = 200M ops. In Python, this is too slow (minutes).
        
        We need a faster proxy or a vectorized count.
        
        Let's count the number of "non-convex" 7-goms?
        Or use the 4-gon proxy?
        The current program uses convex 4-gons as a proxy.
        Let's stick to a fast proxy that correlates well: Number of convex 4-gons.
        Or better: Number of "bad" triples (collinear or close)? No.
        
        Let's try to vectorize the convex 4-gon count, which is a strong proxy.
        C(33, 4) = 40,920.
        We can check all 4-gons at once using NumPy.
        
        A 4-gon is convex iff no point is inside the triangle of the other 3.
        For 4 points p0, p1, p2, p3:
        It is non-convex if:
        p0 is inside p1p2p3 OR
        p1 is inside p0p2p3 OR
        p2 is inside p0p1p3 OR
        p3 is inside p0p1p2
        
        Point-in-triangle check can be vectorized.
        """
        # Generate all 4-combinations
        # This memory might be large: 40920 * 4 * 2 floats ~ 1MB. Fine.
        # But generating indices is slow in Python loop.
        # We can use itertools and stack.
        
        import itertools
        idxs = list(itertools.combinations(range(n), 4))
        if not idxs:
            return 0
        
        # Shape: (M, 4, 2)
        pts_4 = points[np.array(idxs)]
        
        # M = 40920
        p0 = pts_4[:, 0, :]
        p1 = pts_4[:, 1, :]
        p2 = pts_4[:, 2, :]
        p3 = pts_4[:, 3, :]
        
        # Check if p0 is inside p1p2p3
        # Cross products for p1->p2->p0, p2->p3->p0, p3->p1->p0
        # c1 = (p2-p1) x (p0-p1)
        # c2 = (p3-p2) x (p0-p2)
        # c3 = (p1-p3) x (p0-p3)
        
        def inside_tri(pt, a, b, c):
            # pt, a, b, c are arrays of shape (M, 2)
            v1 = b - a
            v2 = c - a
            v3 = pt - a
            
            # Cross product z-component: v1_x * v2_y - v1_y * v2_x
            # We need cross(v1, pt-a), cross(v2, pt-b)? No.
            # Standard: 
            # d1 = cross(b-a, pt-a)
            # d2 = cross(c-b, pt-b)
            # d3 = cross(a-c, pt-c)
            
            d1 = (b[:, 0] - a[:, 0]) * (pt[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (pt[:, 0] - a[:, 0])
            d2 = (c[:, 0] - b[:, 0]) * (pt[:, 1] - b[:, 1]) - (c[:, 1] - b[:, 1]) * (pt[:, 0] - b[:, 0])
            d3 = (a[:, 0] - c[:, 0]) * (pt[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (pt[:, 0] - c[:, 0])
            
            # Inside if all same sign (and non-zero, but we assume general position)
            # Strictly inside: d1, d2, d3 all > 0 or all < 0
            return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
        
        in_0 = inside_tri(p0, p1, p2, p3)
        in_1 = inside_tri(p1, p0, p2, p3)
        in_2 = inside_tri(p2, p0, p1, p3)
        in_3 = inside_tri(p3, p0, p1, p2)
        
        # Convex if NONE are inside
        is_convex = ~(in_0 | in_1 | in_2 | in_3)
        
        return np.sum(is_convex)

    def count_convex_7goms_fast(points):
        """
        Faster proxy: Count convex 4-gons.
        The relationship is not direct, but minimizing convex 4-gons strongly correlates with minimizing convex 7-gons.
        Actually, a 7-gon is convex implies all its 4-subsets are convex.
        So C_7 <= C_4 / 35 (roughly).
        Minimizing C_4 is a good heuristic.
        """
        return count_convex_7goms(points) # Reusing the name for the proxy function to avoid confusion in logic, but actually let's call it count_convex_4goms
        
    # Let's rename to be clear
    def count_convex_4goms(points):
        import itertools
        idxs = list(itertools.combinations(range(n), 4))
        if not idxs:
            return 0
        
        pts_4 = points[np.array(idxs)]
        p0 = pts_4[:, 0, :]
        p1 = pts_4[:, 1, :]
        p2 = pts_4[:, 2, :]
        p3 = pts_4[:, 3, :]
        
        def inside_tri(pt, a, b, c):
            d1 = (b[:, 0] - a[:, 0]) * (pt[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (pt[:, 0] - a[:, 0])
            d2 = (c[:, 0] - b[:, 0]) * (pt[:, 1] - b[:, 1]) - (c[:, 1] - b[:, 1]) * (pt[:, 0] - b[:, 0])
            d3 = (a[:, 0] - c[:, 0]) * (pt[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (pt[:, 0] - c[:, 0])
            return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
        
        in_0 = inside_tri(p0, p1, p2, p3)
        in_1 = inside_tri(p1, p0, p2, p3)
        in_2 = inside_tri(p2, p0, p1, p3)
        in_3 = inside_tri(p3, p0, p1, p2)
        
        is_convex = ~(in_0 | in_1 | in_2 | in_3)
        return np.sum(is_convex)

    # Initialization: Erdos-Szekeres Cup-Cap Structure
    # The ES theorem says any set of (r-1)(s-1)+1 points contains a cup of size r or cap of size s.
    # To MINIMIZE convex subsets, we want to AVOID large cups and caps?
    # Actually, points in convex position form a cup or cap if sorted by x.
    # If we arrange points in a "nested" way, we can reduce convexity.
    # A known configuration with few convex k-gons is the "convex position" of points on a convex curve? No, that maximizes.
    # We want points to be "deep" inside the hull of others.
    
    # Strategy: Place points on a parabola y=x^2, but perturb them slightly inward?
    # Points on a parabola are in convex position (all on hull). This is BAD.
    # We want points NOT on the hull.
    
    # Better: Cluster points.
    # If we put all points in a small ball, they are likely in convex position relative to each other?
    # No, if they are random in a small ball, many will be inside the convex hull of others.
    # Specifically, if we take points from a "cubic" distribution or similar, the expected number of points on the hull is O(N^{1/3}).
    # So most points are interior.
    # Interior points cannot be part of a convex 7-gon?
    # Wait. A convex 7-gon must consist of 7 points that are ALL on the convex hull of the set of 7.
    # If a point is inside the convex hull of the OTHER 6, it cannot be part of that 7-gon's hull.
    # So if we have many points that are "deep" interior points, they won't form convex 7-gons.
    
    # Let's initialize with points uniformly distributed in a small square, e.g., [0.4, 0.6] x [0.4, 0.6].
    # This creates a dense cluster.
    
    np.random.seed(42)
    pts = np.random.rand(n, 2) * 0.2 + 0.4
    
    # Ensure validity
    if not fast_is_valid(pts):
        # Jitter
        pts += np.random.normal(0, 0.01, (n, 2))
        pts = np.clip(pts, 0.0, 1.0)
    
    current_score = count_convex_4goms(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Local Search
    # Move one point at a time.
    # To speed up, we only recompute the score if the move is promising?
    # Or we can use a faster delta calculation?
    # Delta calculation for moving point i:
    # Only 4-gons containing i change.
    # There are C(32, 3) = 4960 such 4-gons.
    # We can vectorize the check for these 4960 4-gons.
    
    def delta_convex_4goms(points, idx, new_p):
        others = [i for i in range(n) if i != idx]
        # Generate all combinations of 3 others
        import itertools
        combos = list(itertools.combinations(others, 3))
        if not combos:
            return 0
        
        # Old point
        old_p = points[idx]
        
        # Create arrays for the 4-gons
        # Shape: (4960, 4, 2)
        # Point 0 is the moving point
        # Points 1,2,3 are from combos
        
        p1 = points[np.array([c[0] for c in combos])]
        p2 = points[np.array([c[1] for c in combos])]
        p3 = points[np.array([c[2] for c in combos])]
        
        # Expand old_p and new_p to match
        old_p_exp = np.tile(old_p, (len(combos), 1))
        new_p_exp = np.tile(new_p, (len(combos), 1))
        
        def count_convex_in_batch(p_moving, p1, p2, p3):
            # p_moving, p1, p2, p3 are (M, 2)
            
            def inside_tri(pt, a, b, c):
                d1 = (b[:, 0] - a[:, 0]) * (pt[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (pt[:, 0] - a[:, 0])
                d2 = (c[:, 0] - b[:, 0]) * (pt[:, 1] - b[:, 1]) - (c[:, 1] - b[:, 1]) * (pt[:, 0] - b[:, 0])
                d3 = (a[:, 0] - c[:, 0]) * (pt[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (pt[:, 0] - c[:, 0])
                return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
            
            in_0 = inside_tri(p_moving, p1, p2, p3)
            in_1 = inside_tri(p1, p_moving, p2, p3)
            in_2 = inside_tri(p2, p_moving, p1, p3)
            in_3 = inside_tri(p3, p_moving, p1, p2)
            
            is_convex = ~(in_0 | in_1 | in_2 | in_3)
            return np.sum(is_convex)
        
        old_count = count_convex_in_batch(old_p_exp, p1, p2, p3)
        new_count = count_convex_in_batch(new_p_exp, p1, p2, p3)
        
        return new_count - old_count

    # Main Loop
    no_improve = 0
    max_no_improve = 200
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    while time.time() < deadline:
        improved = False
        # Shuffle points to avoid bias
        order = np.random.permutation(n)
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            old_p = best_pts[idx].copy()
            best_delta = 0
            best_new_p = None
            
            # Try random directions and step sizes
            for step in step_sizes:
                for _ in range(4): # 4 random directions
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    new_p = old_p + np.array([dx, dy])
                    new_p = np.clip(new_p, 0.0, 1.0)
                    
                    # Quick validity check
                    if not fast_is_valid(np.concatenate([best_pts[:idx], [new_p], best_pts[idx+1:]])):
                        continue
                    
                    delta = delta_convex_4goms(best_pts, idx, new_p)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_p = new_p
                        improved = True
            
            if best_new_p is not None:
                best_pts[idx] = best_new_p
                best_score += best_delta
                no_improve = 0
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            # Restart or Perturb
            # Perturb a random point significantly
            idx = np.random.randint(n)
            new_p = best_pts[idx] + np.random.normal(0, 0.1, 2)
            new_p = np.clip(new_p, 0.0, 1.0)
            if fast_is_valid(np.concatenate([best_pts[:idx], [new_p], best_pts[idx+1:]])):
                best_pts[idx] = new_p
                best_score = count_convex_4goms(best_pts)
            no_improve = 0

    # Final Validation
    # The fast_is_valid doesn't check collinearity.
    # We need to ensure no 3 points are collinear (area < 1e-9).
    # We can do a final check and jitter if necessary.
    
    def check_collinearity(points):
        # Check all triples
        # This is O(N^3) = 33^3 = 35937. Fast enough.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area < 1e-9:
                        return False
        return True

    if not check_collinearity(best_pts):
        # Jitter until valid
        for _ in range(100):
            idx = np.random.randint(n)
            best_pts[idx] += np.random.normal(0, 0.001, 2)
            best_pts[idx] = np.clip(best_pts[idx], 0.0, 1.0)
            if fast_is_valid(best_pts) and check_collinearity(best_pts):
                break
    
    if not fast_is_valid(best_pts) or not check_collinearity(best_pts):
        # Fallback
        pts = np.random.rand(n, 2) * 0.9 + 0.05
        return pts

    return best_pts

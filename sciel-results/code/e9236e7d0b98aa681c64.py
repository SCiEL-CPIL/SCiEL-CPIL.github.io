# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _cross(p, q, r):
    """Cross product (q-p) x (r-p)"""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _count_convex7(points):
    """Count convex 7-gons using the DP over convex chains described in the task."""
    n = len(points)
    if n < 7:
        return 0
    
    # Sort lexicographically
    idx = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[idx]
    
    # For each pivot i, count convex chains of length 7 ending at i
    # A convex chain p0 -> p1 -> ... -> pk is one where each turn is convex
    # and all points are to the same side of the chain direction
    
    count = 0
    
    for pivot in range(n):
        # Points after pivot in lex order
        later = [j for j in range(pivot + 1, n)]
        if len(later) < 6:
            continue
        
        # Sort later points by angle around pivot
        angles = np.arctan2(pts[later, 1] - pts[pivot, 1], pts[later, 0] - pts[pivot, 0])
        order = np.argsort(angles)
        sorted_later = [later[i] for i in order]
        
        # DP: dp[chain_len][last_point] = number of convex chains of length chain_len
        # ending at last_point, starting from pivot
        # We want chains of exactly 7 points including pivot, so 6 more points
        
        # dp[j][q] = number of convex chains of length j+1 (including pivot) ending at q
        # where j ranges from 1 to 6
        dp = np.zeros((7, n), dtype=np.int64)
        
        # Chains of length 1: just the pivot, but we start with pivot -> q
        # So dp[1][q] = 1 for each q in sorted_later (chain: pivot, q)
        for q in sorted_later:
            dp[1][q] = 1
        
        # Build longer chains
        for chain_len in range(2, 7):
            for r_idx, r in enumerate(sorted_later):
                # Try all q before r in the angular order
                for q_idx in range(r_idx):
                    q = sorted_later[q_idx]
                    # Check if turn pivot -> q -> r is convex
                    # (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9
                    turn = (pts[q, 0] - pts[pivot, 0]) * (pts[r, 1] - pts[q, 1]) - \
                           (pts[q, 1] - pts[pivot, 1]) * (pts[r, 0] - pts[q, 0])
                    if turn > 1e-9:
                        dp[chain_len][r] += dp[chain_len - 1][q]
        
        # Add chains of length 7 that close back to pivot
        # For a chain pivot -> p1 -> ... -> p6, we need the closing turn p6 -> pivot to be convex
        # The closing turn: (px - p6x)(py - pivot_y) ... wait, let me think about this more carefully
        
        # Actually, for a convex polygon, all turns including the closing one must be convex
        # The closing turn is p6 -> pivot -> p1
        # Turn at pivot: (p1 - pivot) x (p6 - pivot) should be convex
        # But since we sorted by angle, p1 has the smallest angle and p6 has the largest
        # The turn p6 -> pivot -> p1 is convex if the angle from p6 to p1 going the short way is < pi
        
        # For the DP approach, we need to ensure the polygon is convex
        # Let me use a different approach: enumerate all 7-point subsets and check convexity
        
        # Actually, let me just count directly for small n
        pass
    
    # Direct enumeration for correctness
    for subset in combinations(range(n), 7):
        pts_subset = pts[list(subset)]
        
        # Check if these 7 points form a convex polygon
        # Sort by angle around centroid
        cx = np.mean(pts_subset[:, 0])
        cy = np.mean(pts_subset[:, 1])
        angles = np.arctan2(pts_subset[:, 1] - cy, pts_subset[:, 0] - cx)
        order = np.argsort(angles)
        sorted_pts = pts_subset[order]
        
        # Check all turns are convex (same sign)
        convex = True
        signs = []
        for i in range(7):
            p = sorted_pts[i]
            q = sorted_pts[(i + 1) % 7]
            r = sorted_pts[(i + 2) % 7]
            cross_val = (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])
            signs.append(cross_val)
        
        # All must be positive or all negative
        if all(s > 1e-9 for s in signs) or all(s < -1e-9 for s in signs):
            count += 1
    
    return count


def _is_valid(points):
    """Check if points are valid per the task contract."""
    n = len(points)
    if n != 33:
        return False
    if not np.all(np.isfinite(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    
    # Check minimum distance
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            if dx * dx + dy * dy < 1e-12:
                return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                val = (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) - \
                      (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                if abs(val) < 1e-9:
                    return False
    
    return True


def _generate_cup_cap(n=33):
    """Generate a cup-cap construction: points on two convex chains (lower and upper)."""
    points = []
    
    # Lower chain (cup): points on a convex curve, say y = x^2
    n_lower = (n + 1) // 2  # 17 points
    n_upper = n - n_lower    # 16 points
    
    # Lower chain: from left to right, convex downward
    for i in range(n_lower):
        t = i / (n_lower - 1) if n_lower > 1 else 0
        x = 0.05 + 0.9 * t
        y = 0.05 + 0.15 * t * t  # Gentle upward curve
        points.append([x, y])
    
    # Upper chain: from right to left, convex upward
    for i in range(n_upper):
        t = i / (n_upper - 1) if n_upper > 1 else 0
        x = 0.95 - 0.9 * t
        y = 0.85 + 0.1 * t * t  # Gentle upward curve
        points.append([x, y])
    
    return np.array(points)


def _local_search(points, max_time=300, seed=42):
    """Local search to minimize convex 7-gons."""
    rng = np.random.RandomState(seed)
    n = len(points)
    
    # Initial count
    best_points = points.copy()
    best_count = _count_convex7(best_points)
    
    start_time = time.time()
    timeout = max_time
    
    # Hill climbing with random perturbations
    step_size = 0.02
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < timeout and no_improve < max_no_improve:
        # Try a random perturbation
        i = rng.randint(n)
        delta = rng.normal(0, step_size, size=2)
        
        new_points = best_points.copy()
        new_points[i] += delta
        
        # Clip to unit square
        new_points = np.clip(new_points, 1e-4, 1 - 1e-4)
        
        # Check validity
        if not _is_valid(new_points):
            no_improve += 1
            continue
        
        # Count convex 7-gons
        new_count = _count_convex7(new_points)
        
        if new_count < best_count:
            best_points = new_points
            best_count = new_count
            no_improve = 0
            # Reduce step size slightly
            step_size = max(0.001, step_size * 0.95)
        else:
            no_improve += 1
    
    return best_points, best_count


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 2^(k-2) + 1 points in general
    position (no three collinear) inside the unit square so as to MINIMIZE the number of
    convex k-gons whose vertices are among the points. Here k = 7, so n = 2^5 + 1 = 33.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points,
            each lying in the unit square [0, 1] x [0, 1].
    """
    n = 33
    
    # Start with cup-cap construction
    points = _generate_cup_cap(n)
    
    # Ensure validity with small perturbations if needed
    if not _is_valid(points):
        np.random.seed(0)
        points += np.random.randn(n, 2) * 1e-4
        points = np.clip(points, 1e-4, 1 - 1e-4)
    
    # Local search to minimize convex 7-gons
    deadline = time.time() + 600
    search_time = min(300, max(10, deadline - time.time() - 60))
    
    best_points, best_count = _local_search(points, max_time=search_time, seed=42)
    
    # Final validity check
    if not _is_valid(best_points):
        # Fallback to original
        best_points = points
    
    return best_points


# EVOLVE-BLOCK-END

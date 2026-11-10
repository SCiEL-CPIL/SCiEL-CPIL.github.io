# EVOLVE-BLOCK-START
import numpy as np
import time


def _count_convex_7gons(points):
    """Count number of 7-point subsets in convex position using the specified DP."""
    n = len(points)
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    # For each pivot i, consider points j > i
    # Sort by angle around pivot
    # Build convex chains
    
    # For 7-gons: we need 7 points in convex position
    # A point set is in convex position if they form a convex polygon
    
    # DP approach: 
    # For each pivot p_i, look at points with index > i
    # Sort by angle
    # A convex chain from p_i through k points
    # A 7-gon with p_i as the "lowest" point (in lex order) 
    # consists of two convex chains from p_i to some p_j, with j > i,
    # and the total number of intermediate points on both chains = 5
    
    # Let's use the standard approach:
    # For pivot i, let S_i = {j : j > i}
    # Sort S_i by atan2 angle around p_i
    # A convex chain is a sequence i = v_0, v_1, ..., v_m such that 
    # each turn is convex (left turn in the angle-sorted order)
    # 
    # Actually, the turn condition: p -> q -> r is convex if 
    # (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9
    # This is the cross product (q-p) x (r-q) > 0, meaning left turn
    
    # For a convex 7-gon with minimum lex point being p_i:
    # The other 6 points form a convex chain from the next point back to p_i
    # Actually, in the angle-sorted order around p_i, the 6 other points
    # must form a convex polygon with p_i.
    
    # Simpler: brute force check for each 7-subset? C(33,7) = 5,379,616 which is feasible.
    
    count = 0
    # For each combination of 7 points, check if they are in convex position
    # A set of points is in convex position if all points are vertices of their convex hull
    # Equivalently, when sorted by angle around centroid, all turns are same direction
    
    # More efficient: for a 7-point set, sort by angle around centroid,
    # check if all cross products have same sign
    
    from itertools import combinations
    
    for combo in combinations(range(n), 7):
        sub = pts[list(combo)]
        # Compute centroid
        cx = sub[:, 0].mean()
        cy = sub[:, 1].mean()
        # Angles
        angles = np.arctan2(sub[:, 1] - cy, sub[:, 0] - cx)
        # Sort by angle
        order_sub = np.argsort(angles)
        sorted_sub = sub[order_sub]
        
        # Check if all turns are convex (same direction)
        # Compute cross products of consecutive edges
        all_pos = True
        all_neg = True
        for k in range(7):
            p1 = sorted_sub[k]
            p2 = sorted_sub[(k+1) % 7]
            p3 = sorted_sub[(k+2) % 7]
            cross = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
            if cross < -1e-9:
                all_pos = False
            if cross > 1e-9:
                all_neg = False
        if all_pos or all_neg:
            count += 1
    
    return count


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Minimize convex 7-gons among 33 points in unit square.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Start with a structured initial guess: points on a parabola (concave down)
    # This tends to have fewer convex k-gons than random
    # Actually, let's try points on two lines (double chain)
    
    # Initial: points arranged in a "V" shape or on a concave curve
    # Try: y = 1 - x^2 (concave down) with small perturbations
    np.random.seed(42)
    
    # Start with points on a concave curve
    x = np.linspace(0.05, 0.95, n)
    y = 1.0 - x**2
    # Add small random perturbation to break collinearity
    x = x + np.random.uniform(-0.01, 0.01, n)
    y = y + np.random.uniform(-0.01, 0.01, n)
    # Clip to [0, 1]
    x = np.clip(x, 1e-4, 1 - 1e-4)
    y = np.clip(y, 1e-4, 1 - 1e-4)
    
    points = np.column_stack([x, y])
    
    # Verify validity
    def _is_valid(pts):
        if pts.shape != (33, 2):
            return False
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return False
        if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
            return False
        # Check pairwise distances
        for i in range(33):
            for j in range(i+1, 33):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                if dx*dx + dy*dy < 1e-12:
                    return False
        # Check no three collinear
        for i in range(33):
            for j in range(i+1, 33):
                for k in range(j+1, 33):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    val = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if val < 1e-9:
                        return False
        return True
    
    if not _is_valid(points):
        # Fallback to random
        np.random.seed(0)
        points = np.random.rand(33, 2)
    
    best_points = points.copy()
    best_count = _count_convex_7gons(points)
    best_value = 1.0 / (1.0 + best_count)
    
    # Local search: try perturbing individual points
    step_size = 0.05
    no_improve = 0
    
    while time.time() < deadline - 60:  # 60s safety margin
        # Try perturbing one point
        i = np.random.randint(n)
        old_x, old_y = points[i]
        points[i, 0] += np.random.uniform(-step_size, step_size)
        points[i, 1] += np.random.uniform(-step_size, step_size)
        points[i, 0] = np.clip(points[i, 0], 1e-4, 1 - 1e-4)
        points[i, 1] = np.clip(points[i, 1], 1e-4, 1 - 1e-4)
        
        if not _is_valid(points):
            points[i, 0] = old_x
            points[i, 1] = old_y
            continue
        
        count = _count_convex_7gons(points)
        value = 1.0 / (1.0 + count)
        
        if value > best_value:
            best_value = value
            best_count = count
            best_points = points.copy()
            no_improve = 0
            # Decrease step size slightly
            step_size *= 0.95
        else:
            points[i, 0] = old_x
            points[i, 1] = old_y
            no_improve += 1
        
        if no_improve > 100:
            # Reset to best and try larger step
            points = best_points.copy()
            step_size = 0.05
            no_improve = 0
    
    return best_points


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _is_valid(points):
    """Check validity constraints."""
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check pairwise squared distances
    for i in range(33):
        for j in range(i + 1, 33):
            d2 = (points[i, 0] - points[j, 0])**2 + (points[i, 1] - points[j, 1])**2
            if d2 < 1e-12:
                return False
    # Check no three collinear
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                val = abs(points[i, 0] * (points[j, 1] - points[k, 1]) +
                          points[j, 0] * (points[k, 1] - points[i, 1]) +
                          points[k, 0] * (points[i, 1] - points[j, 1]))
                if val < 1e-9:
                    return False
    return True


def _count_convex_7gons(points):
    """
    Count the number of 7-point subsets in convex position using a dynamic program
    over convex chains as described in the task contract.
    
    Points are sorted lexicographically. For each pivot, later points are sorted
    by atan2 angle around it. A turn p -> q -> r counts as convex only if
    (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9, including closing turns back to pivot.
    """
    n = len(points)
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    C = 0
    for i in range(n - 6):
        pivot = pts[i]
        # Later points
        later = pts[i + 1:]
        # Sort by atan2 angle around pivot
        angles = np.arctan2(later[:, 1] - pivot[1], later[:, 0] - pivot[0])
        sorted_later = later[np.argsort(angles)]
        m = len(sorted_later)
        if m < 6:
            continue
        
        # dp[l][i][j] = number of valid chains of length l (l points from sorted_later)
        # where the last two points are at indices i and j (i < j), and all turns are valid.
        # 
        # l=2: dp[2][i][j] = 1 if turn(pivot, sl[i], sl[j]) is convex, else 0.
        # l>=3: dp[l][i][j] = sum over h < i of dp[l-1][h][i] * (1 if turn(sl[h], sl[i], sl[j]) is convex else 0)
        #
        # Final count for pivot: sum over i2 < j2 of dp[6][i2][j2] * (1 if turn(sl[j2], pivot, sl[i2]) is convex else 0)
        
        if m < 6:
            continue
            
        # Use 2D arrays for dp
        # dp[l] is a 2D array of shape (m, m), where dp[l][i][j] is defined for i < j
        # We'll use a list of 2D arrays, one for each l from 2 to 6
        
        # Precompute turn values for efficiency
        # turn[p][q][r] = (qx-px)(ry-qy) - (qy-py)(rx-qx) for p, q, r in sorted_later
        # But we also need turns involving pivot.
        
        # Let's just compute on the fly but use numpy for vectorization where possible.
        
        # dp[2][i][j] for 0 <= i < j < m
        dp2 = np.zeros((m, m), dtype=np.int64)
        for j in range(1, m):
            for i in range(j):
                # turn(pivot, sl[i], sl[j])
                cross = (sorted_later[i, 0] - pivot[0]) * (sorted_later[j, 1] - sorted_later[i, 1]) - \
                        (sorted_later[i, 1] - pivot[1]) * (sorted_later[j, 0] - sorted_later[i, 0])
                if cross > 1e-9:
                    dp2[i, j] = 1
        
        # Build up to l=6
        dp = [None] * 7  # dp[l] for l in 2..6
        dp[2] = dp2
        
        for l in range(3, 7):
            dp[l] = np.zeros((m, m), dtype=np.int64)
            for j in range(2, m):
                for i in range(1, j):
                    total = 0
                    for h in range(i):
                        prev_val = dp[l-1][h, i]
                        if prev_val == 0:
                            continue
                        # turn(sl[h], sl[i], sl[j])
                        cross = (sorted_later[i, 0] - sorted_later[h, 0]) * (sorted_later[j, 1] - sorted_later[i, 1]) - \
                                (sorted_later[i, 1] - sorted_later[h, 1]) * (sorted_later[j, 0] - sorted_later[i, 0])
                        if cross > 1e-9:
                            total += prev_val
                    dp[l][i, j] = total
        
        # Now sum up valid 6-point chains with closing turn
        for j in range(5, m):
            for i in range(4, j):
                val = dp[6][i, j]
                if val == 0:
                    continue
                # Check closing turn: sl[j] -> pivot -> sl[i]
                # p = sl[j], q = pivot, r = sl[i]
                cross = (pivot[0] - sorted_later[j, 0]) * (sorted_later[i, 1] - pivot[1]) - \
                        (pivot[1] - sorted_later[j, 1]) * (sorted_later[i, 0] - sorted_later[j, 0])
                if cross > 1e-9:
                    C += val
    
    return C


def _local_search(points, count, deadline, max_steps=50):
    """
    Local search: try small perturbations of each point to reduce convex 7-gon count.
    """
    n = len(points)
    current = points.copy()
    current_count = count
    step = 0.02
    
    for _ in range(max_steps):
        if time.time() > deadline - 30:
            break
        
        improved = False
        # Try perturbing each point in 4 directions
        for i in range(n):
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                new_pts = current.copy()
                new_pts[i, 0] += dx
                new_pts[i, 1] += dy
                
                # Quick bounds check
                if new_pts[i, 0] < 0 or new_pts[i, 0] > 1 or new_pts[i, 1] < 0 or new_pts[i, 1] > 1:
                    continue
                
                # Quick validity check (just the modified point's constraints)
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    d2 = (new_pts[i, 0] - new_pts[j, 0])**2 + (new_pts[i, 1] - new_pts[j, 1])**2
                    if d2 < 1e-12:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check no three collinear involving point i
                for j in range(n):
                    if j == i:
                        continue
                    for k in range(j + 1, n):
                        if k == i:
                            continue
                        val = abs(new_pts[i, 0] * (new_pts[j, 1] - new_pts[k, 1]) +
                                  new_pts[j, 0] * (new_pts[k, 1] - new_pts[i, 1]) +
                                  new_pts[k, 0] * (new_pts[i, 1] - new_pts[j, 1]))
                        if val < 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Count convex 7-gons
                new_count = _count_convex_7gons(new_pts)
                
                if new_count < current_count:
                    current = new_pts
                    current_count = new_count
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 0.001:
                break
    
    return current, current_count


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place 33 points in the unit square
    to minimize the number of convex 7-gons.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin (630s limit, leave 60s)
    
    best_points = None
    best_count = float('inf')
    
    np.random.seed(42)
    
    n = 33
    
    # Phase 1: Random restarts with local search
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 60:
            break
        
        # Generate random points
        points = np.random.rand(n, 2)
        
        # Quick validity check
        if not _is_valid(points):
            continue
        
        # Count convex 7-gons
        count = _count_convex_7gons(points)
        
        # Local search
        points, count = _local_search(points, count, deadline, max_steps=30)
        
        if count < best_count:
            best_count = count
            best_points = points.copy()
        
        # If we found 0, we can stop
        if best_count == 0:
            break
    
    # Phase 2: Try structured constructions
    if time.time() < deadline - 30:
        # Try points on a parabola (convex curve) - should have fewer convex polygons
        # Actually, points on a convex curve will have MANY convex polygons.
        # We want points that are "clustered" or on a concave curve.
        
        # Try points on a concave curve (e.g., y = x^2 but flipped, or a circle arc)
        # Points on a circle: all points are in convex position, so C(33,7) convex 7-gons. Bad.
        
        # Try a "double spiral" or "S-curve" construction
        t = np.linspace(0, 1, n)
        # S-curve: x = t, y = 0.5 + 0.4 * np.sin(np.pi * t)
        # This creates a curve that goes up then down, reducing convex subsets.
        x = t
        y = 0.5 + 0.4 * np.sin(np.pi * t)
        points = np.column_stack([x, y])
        
        # Add small random perturbation to avoid collinearity
        np.random.seed(123)
        points += np.random.normal(0, 0.005, (n, 2))
        
        # Clamp to [0, 1]
        points = np.clip(points, 0, 1)
        
        if _is_valid(points):
            count = _count_convex_7gons(points)
            points, count = _local_search(points, count, deadline, max_steps=20)
            if count < best_count:
                best_count = count
                best_points = points.copy()
    
    if time.time() < deadline - 30:
        # Try a "two-cluster" construction: 16 points near (0.25, 0.5) and 17 near (0.75, 0.5)
        np.random.seed(456)
        pts1 = np.random.normal((0.25, 0.5), 0.05, (16, 2))
        pts2 = np.random.normal((0.75, 0.5), 0.05, (17, 2))
        points = np.vstack([pts1, pts2])
        points = np.clip(points, 0, 1)
        
        if _is_valid(points):
            count = _count_convex_7gons(points)
            points, count = _local_search(points, count, deadline, max_steps=20)
            if count < best_count:
                best_count = count
                best_points = points.copy()
    
    if time.time() < deadline - 30:
        # Try a "grid with perturbation" construction
        # 6x6 grid = 36 points, take 33
        rows, cols = 6, 6
        grid_pts = []
        for i in range(rows):
            for j in range(cols):
                grid_pts.append([j / (cols - 1), i / (rows - 1)])
        grid_pts = np.array(grid_pts[:33])
        
        np.random.seed(789)
        grid_pts += np.random.normal(0, 0.01, (33, 2))
        grid_pts = np.clip(grid_pts, 0, 1)
        
        if _is_valid(grid_pts):
            count = _count_convex_7gons(grid_pts)
            grid_pts, count = _local_search(grid_pts, count, deadline, max_steps=20)
            if count < best_count:
                best_count = count
                best_points = grid_pts.copy()
    
    if best_points is None:
        # Fallback
        np.random.seed(0)
        best_points = np.random.rand(n, 2)
    
    return best_points


# EVOLVE-BLOCK-END

import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 10.0

def _count_convex4_fast(points):
    """
    Count the number of convex quadrilaterals (4-gons) in the set of points.
    A set of 4 points is a convex quadrilateral if and only if they are not in convex position 
    with one point inside the triangle of the other three.
    Equivalently, for 4 points in general position, they form a convex quad if and only if
    the sum of the "left turns" around the perimeter is 360 degrees, or more simply:
    Sort the 4 points lexicographically. 
    Actually, a simpler check: 
    4 points form a convex quad iff no point lies inside the triangle formed by the other 3.
    For 4 points A, B, C, D:
    They are convex if and only if:
    1. The points are in convex position.
    This happens if the origin is inside the triangle of the other 3? No.
    
    Efficient check for 4 points:
    Let the points be p0, p1, p2, p3.
    They form a convex quadrilateral if and only if the sum of the cross products of consecutive edges (in angular order around the centroid) is positive (all same sign) and non-zero.
    Or simpler:
    Check if any point is inside the triangle of the other three.
    If none is inside, it is convex.
    
    Even faster:
    For 4 points, there are 3 ways to partition them into two pairs.
    If the segments connecting the pairs intersect, it's a convex quad? No, that's for the diagonals.
    
    Let's use the property:
    4 points are convex if and only if the number of "left turns" in the sorted-by-angle order around the centroid is 4 (or -4).
    Or:
    4 points are NOT convex (i.e., one is inside the triangle of the other 3) if and only if the origin is inside the triangle of the other 3? No.
    
    Standard O(1) check for 4 points:
    Let points be A, B, C, D.
    They are convex if and only if:
    sign(cross(B-A, C-B)) == sign(cross(C-B, D-C)) == sign(cross(D-C, A-D)) == sign(cross(A-D, B-A))
    This assumes the correct cyclic order.
    
    Robust O(1) method:
    1. Compute the convex hull of the 4 points.
    2. If the hull has 4 vertices, it's convex.
    3. If the hull has 3 vertices, it's not.
    Convex hull of 4 points is fast.
    
    Alternatively:
    4 points are convex if and only if the sum of the angles around the centroid is 2*pi.
    
    Let's use the "inside triangle" check.
    Point P is inside triangle ABC if P and A are on the same side of BC, P and B are on the same side of AC, P and C are on the same side of AB.
    So, for 4 points, they are NOT convex if and only if one of the 4 points is inside the triangle of the other 3.
    This requires 4 triangle containment checks.
    
    def point_in_triangle(p, a, b, c):
        d1 = cross(p-a, b-a)
        d2 = cross(p-b, c-b)
        d3 = cross(p-c, a-c)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)
        
    This is 4 * 3 cross products = 12 cross products per 4-subset.
    Number of 4-subsets in 33 points: 33C4 = 40920.
    Total operations: 40920 * 12 ~ 500,000. This is very fast in Python if vectorized or even in a loop.
    """
    n = len(points)
    count = 0
    # We can vectorize this heavily.
    # Generate all indices
    # But 40920 is small enough for a loop in Python if we optimize the inner logic.
    # However, let's try to vectorize the "is convex" check for all 4-subsets at once?
    # That would require a huge array.
    
    # Let's just loop. 40k iterations is trivial.
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    p0 = points[i]
                    p1 = points[j]
                    p2 = points[k]
                    p3 = points[l]
                    
                    # Check if any point is inside the triangle of the other three
                    # If none is inside, it's convex.
                    is_convex = True
                    # Check if p0 is inside (p1, p2, p3)
                    d1 = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
                    d2 = (p2[0]-p0[0])*(p3[1]-p0[1]) - (p2[1]-p0[1])*(p3[0]-p0[0])
                    d3 = (p3[0]-p0[0])*(p1[1]-p0[1]) - (p3[1]-p0[1])*(p1[0]-p0[0])
                    # Wait, this checks if p0 is inside p1,p2,p3? No, this checks signs relative to p0.
                    # The standard check:
                    # P is inside ABC if cross(A,P,B) and cross(B,P,C) and cross(C,P,A) have the same sign.
                    # Let's use that.
                    
                    def is_inside(p, a, b, c):
                        c1 = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                        c2 = (c[0]-b[0])*(p[1]-b[1]) - (c[1]-b[1])*(p[0]-b[0])
                        c3 = (a[0]-c[0])*(p[1]-c[1]) - (a[1]-c[1])*(p[0]-c[0])
                        has_neg = (c1 < 0) or (c2 < 0) or (c3 < 0)
                        has_pos = (c1 > 0) or (c2 > 0) or (c3 > 0)
                        return not (has_neg and has_pos)

                    if is_inside(p0, p1, p2, p3): is_convex = False
                    elif is_inside(p1, p0, p2, p3): is_convex = False
                    elif is_inside(p2, p0, p1, p3): is_convex = False
                    elif is_inside(p3, p0, p1, p2): is_convex = False
                    
                    if is_convex:
                        count += 1
    return count

def _count_convex7_fast(points):
    """
    Count convex 7-gons using the DP method from the seed, but optimized.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue
        
        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]
        
        # dp[l, s, p_idx, c_idx]
        # l: 2..6
        # s: start index
        # p_idx: previous index
        # c_idx: current index
        # Size: 5 * m * m * m. For m=32, 5 * 32768 * 32? No, m^3 is 32768. 5*32768 = 163k.
        # This is small.
        
        dp = np.zeros((7, m, m, m), dtype=np.int64)
        
        # Init l=2
        for s in range(m):
            for c in range(s + 1, m):
                v1x = q[s, 0] - p[0]
                v1y = q[s, 1] - p[1]
                v2x = q[c, 0] - q[s, 0]
                v2y = q[c, 1] - q[s, 1]
                cross = v1x * v2y - v1y * v2x
                if cross > 1e-9:
                    dp[2, s, s, c] = 1
        
        for l in range(3, 7):
            for s in range(m):
                for p_idx in range(s + 1, m):
                    for c_idx in range(p_idx + 1, m):
                        s_val = 0
                        for k in range(s, p_idx):
                            val = dp[l-1, s, k, p_idx]
                            if val == 0:
                                continue
                            v1x = q[p_idx, 0] - q[k, 0]
                            v1y = q[p_idx, 1] - q[k, 1]
                            v2x = q[c_idx, 0] - q[p_idx, 0]
                            v2y = q[c_idx, 1] - q[p_idx, 1]
                            cross = v1x * v2y - v1y * v2x
                            if cross > 1e-9:
                                s_val += val
                        dp[l, s, p_idx, c_idx] = s_val
                        
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp[6, s, p_idx, c_idx]
                    if count == 0:
                        continue
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= 1e-9:
                        continue
                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= 1e-9:
                        continue
                    total += count
    return total

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False
        
    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True

def erdos_szekeres_problem7() -> np.ndarray:
    start = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start)

    # Initial best: Grid-like structure
    # 6x6 grid = 36 points. Take 33.
    # Spacing 0.15. Center 0.5.
    # x in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]?
    # Let's use 0.1 to 0.9.
    # 10x10 grid? No, 33 points.
    # Let's use a 6x6 grid, remove 3 corners.
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 6)
    grid_pts = []
    for x in xs:
        for y in ys:
            grid_pts.append([x, y])
    grid_pts = np.array(grid_pts)
    # Remove 3 points to get 33
    # Remove corners to keep it compact? Or random?
    # Let's just take the first 33.
    best_pts = grid_pts[:33].copy()
    
    # Add jitter to break collinearity
    best_pts += rng.normal(0, 0.02, (33, 2))
    best_pts = np.clip(best_pts, 0.01, 0.99)
    
    if not _is_valid_fast(best_pts):
        # Fallback to random if grid fails validity (unlikely with jitter)
        best_pts = rng.random((33, 2)) * 0.9 + 0.05
        best_pts += rng.normal(0, 0.01, (33, 2))
        best_pts = np.clip(best_pts, 0.01, 0.99)

    # Evaluate true score
    best_c7 = _count_convex7_fast(best_pts)
    # Evaluate proxy score
    best_c4 = _count_convex4_fast(best_pts)

    # Local Search minimizing convex 4-gons
    # Since c4 is much faster, we can do many more iterations.
    # We will check c7 only if c4 improves.
    
    while time_left() > 30:
        improved = False
        # Try moving each point
        for i in range(33):
            if time_left() < 30:
                break
            # Try several random moves for point i
            for _ in range(10):
                if time_left() < 30:
                    break
                new_pts = best_pts.copy()
                # Random perturbation
                new_pts[i] += rng.normal(0, 0.05, 2)
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
                
                if not _is_valid_fast(new_pts):
                    continue
                
                c4_new = _count_convex4_fast(new_pts)
                
                if c4_new < best_c4:
                    # Check true score
                    c7_new = _count_convex7_fast(new_pts)
                    if c7_new < best_c7:
                        best_c7 = c7_new
                        best_c4 = c4_new
                        best_pts = new_pts
                        improved = True
                        break # Restart outer loop if improved significantly
                    # If c4 improved but c7 didn't, we might still accept it if we think c4 is a good proxy
                    # But to be safe, we only accept if c7 improves or is equal?
                    # Actually, if c4 improves, it's a good sign. But we want to minimize c7.
                    # Let's accept if c7 <= best_c7.
                    elif c7_new <= best_c7:
                        best_c4 = c4_new
                        best_pts = new_pts
                        improved = True
                        break
        
        if not improved:
            # Try larger jumps
            for _ in range(50):
                if time_left() < 30:
                    break
                i = rng.integers(0, 33)
                new_pts = best_pts.copy()
                new_pts[i] = rng.random(2) * 0.9 + 0.05
                
                if not _is_valid_fast(new_pts):
                    continue
                
                c4_new = _count_convex4_fast(new_pts)
                if c4_new < best_c4:
                    c7_new = _count_convex7_fast(new_pts)
                    if c7_new <= best_c7:
                        best_c7 = c7_new
                        best_c4 = c4_new
                        best_pts = new_pts
                        improved = True
                        break

    return best_pts

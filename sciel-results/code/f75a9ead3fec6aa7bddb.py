# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    Points are sorted lexicographically, for each pivot the later points are
    sorted by atan2 angle around it, and a turn p -> q -> r counts as convex
    only if (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
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

        # dp[l][s][p_idx][c_idx]
        # l: length of chain (number of points from q)
        # s: start index
        # p_idx: previous index
        # c_idx: current index
        # We only care about l up to 6.
        
        # Initialize dp[2]
        # dp[2][s][s][c] = 1 if turn(p, q[s], q[c]) > 0
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        dp_curr = np.zeros((m, m, m), dtype=np.int64)

        for s in range(m):
            for c in range(s + 1, m):
                v1x = q[s, 0] - p[0]
                v1y = q[s, 1] - p[1]
                v2x = q[c, 0] - q[s, 0]
                v2y = q[c, 1] - q[s, 1]
                cross = v1x * v2y - v1y * v2x
                if cross > 1e-9:
                    dp_prev[s, s, c] = 1

        # Fill dp[3..6]
        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    for c_idx in range(p_idx + 1, m):
                        s_val = 0
                        for k in range(s, p_idx):
                            val = dp_prev[s, k, p_idx]
                            if val == 0:
                                continue
                            # Check turn(q[k], q[p_idx], q[c_idx])
                            v1x = q[p_idx, 0] - q[k, 0]
                            v1y = q[p_idx, 1] - q[k, 1]
                            v2x = q[c_idx, 0] - q[p_idx, 0]
                            v2y = q[c_idx, 1] - q[p_idx, 1]
                            cross = v1x * v2y - v1y * v2x
                            if cross > 1e-9:
                                s_val += val
                        dp_curr[s, p_idx, c_idx] = s_val
            dp_prev, dp_curr = dp_curr, dp_prev

        # Sum up results for l=6
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue

                    # Check closing 1: (q[p_idx], q[c_idx], p)
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= 1e-9:
                        continue

                    # Check closing 2: (q[c_idx], p, q[s])
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
    # Vectorized collinearity check is tricky for all triples, 
    # but n=33 is small enough for a loop if optimized, or we can sample.
    # However, the problem requires ALL triples to be non-collinear.
    # We must check all.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True

def _generate_zigzag(rng):
    """
    Generate points in a zig-zag pattern.
    This keeps the points 'thin' in one dimension, reducing convex subsets.
    """
    xs = np.linspace(0.1, 0.9, 33)
    ys = np.zeros(33)
    # Alternate up and down
    for i in range(33):
        ys[i] = 0.5 + 0.1 * (1 if i % 2 == 0 else -1)
    
    pts = np.column_stack([xs, ys])
    # Add small noise to break exact symmetries and collinearities
    pts += rng.normal(0, 0.01, (33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 50:
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts

def _generate_grid(rng):
    """
    Generate points in a grid pattern.
    """
    # 6x6 grid is 36 points. We need 33.
    # Let's use a 6x6 grid and remove 3 random points.
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 6)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Remove 3 random points
    idx_to_remove = rng.choice(36, size=3, replace=False)
    mask = np.ones(36, dtype=bool)
    mask[idx_to_remove] = False
    pts = pts[mask]
    
    # Add noise
    pts += rng.normal(0, 0.01, (33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 50:
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts

def _generate_clusters(rng):
    """
    Generate points in 4 corner clusters.
    """
    pts = []
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        for _ in range(8):  # 32 points
            x = cx + rng.normal(0, 0.03)
            y = cy + rng.normal(0, 0.03)
            pts.append((x, y))
    # 1 point in center
    x = 0.5 + rng.normal(0, 0.02)
    y = 0.5 + rng.normal(0, 0.02)
    pts.append((x, y))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 50:
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts

def _hill_climb(pts, rng, time_limit):
    """
    Perform hill climbing to minimize the number of convex 7-gons.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 200
    step_size = 0.1
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        # Try moving 1 point
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        
        # Random new position within a neighborhood
        # Occasionally jump to a random location to escape local minima
        if rng.random() < 0.1:
            new_pts[i] = rng.random(2) * 0.8 + 0.1
        else:
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                step_size *= 0.95
            elif c == best_c and best_c > 1:
                # Accept neutral moves to explore
                if rng.random() < 0.3:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # If stuck, try a larger jump
        if not improved and no_improve % 10 == 0:
            n_move = rng.integers(1, 4)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                offset = rng.normal(0, step_size * 2, 2)
                new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size *= 0.9
        
        # Occasionally increase step size to escape local minima
        if no_improve % 50 == 0:
            step_size = min(0.3, step_size * 1.5)
    
    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Generate initial points using multiple strategies
    candidates = []
    
    # Strategy 1: Zig-zag
    pts1 = _generate_zigzag(rng)
    c1 = _count_convex7_fast(pts1)
    candidates.append((c1, pts1))
    
    # Strategy 2: Grid
    pts2 = _generate_grid(rng)
    c2 = _count_convex7_fast(pts2)
    candidates.append((c2, pts2))
    
    # Strategy 3: Clusters
    pts3 = _generate_clusters(rng)
    c3 = _count_convex7_fast(pts3)
    candidates.append((c3, pts3))
    
    # Pick the best candidate
    best_c, best_pts = min(candidates, key=lambda x: x[0])
    
    print(f"Best Initial C: {best_c}")
    
    # Perform hill climbing on the best candidate
    # We will run hill climbing until time is almost up
    # We can also restart if we get stuck, but for now, let's just climb.
    
    # Allocate most of the time to the hill climb
    # We check time left periodically
    while time_left() > 30:
        # Run hill climb for a chunk of time
        chunk_time = min(60, time_left() - 30)
        best_pts, best_c = _hill_climb(best_pts, rng, chunk_time)
        print(f"After hill climb: C={best_c}, t={time.time()-start_time:.1f}")
        
        if best_c <= 1:
            break
        
        # If we have time, try a restart from a slightly perturbed version
        if time_left() > 60 and rng.random() < 0.2:
            perturbed = best_pts + rng.normal(0, 0.05, (33, 2))
            perturbed = np.clip(perturbed, 0.01, 0.99)
            if _is_valid_fast(perturbed):
                c_pert = _count_convex7_fast(perturbed)
                if c_pert < best_c:
                    best_c = c_pert
                    best_pts = perturbed
                    print(f"Restart improved to C: {best_c}")
    
    return best_pts
# EVOLVE-BLOCK-END

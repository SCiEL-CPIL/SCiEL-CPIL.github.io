import time
import numpy as np
from itertools import combinations

_DEADLINE = 625.0
_SAFE_MARGIN = 5.0

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    # Distance check
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    # Collinearity check
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

def _count_convex7_fast(points):
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    eps = 1e-9

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

        # turn_mat[a, b, c] = cross product of (q[b]-q[a]) and (q[c]-q[b])
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        dp_curr = np.zeros((m, m, m), dtype=np.int64)

        # Initialize dp[2]
        # dp[2][s, s, e] = 1 if turn(p, q[s], q[e]) > eps
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:] > eps)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    if not np.any(dp_prev[s, s:p_idx, p_idx] > 0):
                        continue
                        
                    turn_slice = turn_mat[s:p_idx, p_idx, p_idx+1:]
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    mask = turn_slice > eps
                    contrib = (dp_slice[:, np.newaxis] * mask).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            dp_prev, dp_curr = dp_curr, dp_prev

        for s in range(m):
            for c_idx in range(s + 1, m):
                v1x = p[0] - q[c_idx, 0]
                v1y = p[1] - q[c_idx, 1]
                v2x = q[s, 0] - p[0]
                v2y = q[s, 1] - p[1]
                cross_close = v1x * v2y - v1y * v2x
                if cross_close <= eps:
                    continue
                
                for prev in range(s + 1, c_idx):
                    count = dp_prev[s, prev, c_idx]
                    if count == 0:
                        continue
                    v1x = q[c_idx, 0] - q[prev, 0]
                    v1y = q[c_idx, 1] - q[prev, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross = v1x * v2y - v1y * v2x
                    if cross > eps:
                        total += count

    return total

def _make_concentric_polygons(rng):
    """
    Create points on concentric regular polygons.
    Sizes: 6, 6, 6, 6, 6, 3 (Total 33)
    """
    sizes = [6, 6, 6, 6, 6, 3]
    # Radii: increasing from center
    # Center at 0.5, 0.5
    # Max radius 0.45 to stay in [0.05, 0.95]
    # We want them well separated to ensure "nested" property
    radii = [0.05, 0.12, 0.20, 0.28, 0.36, 0.44]
    
    pts = []
    for i, n in enumerate(sizes):
        r = radii[i]
        # Add a random phase shift for each ring to avoid alignment
        phase = rng.uniform(0, 2 * np.pi)
        for j in range(n):
            angle = phase + 2 * np.pi * j / n
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            # Small perturbation to break exact symmetry/collinearity
            x += rng.normal(0, 0.001)
            y += rng.normal(0, 0.001)
            pts.append((x, y))
            
    return np.array(pts)

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Generate initial set of candidates from concentric polygons with different phase shifts
    candidates = []
    for _ in range(20):
        pts = _make_concentric_polygons(rng)
        pts = np.clip(pts, 0.01, 0.99)
        if _is_valid_fast(pts):
            candidates.append(pts)
            
    if not candidates:
        # Fallback to random if concentric fails (unlikely)
        pts = rng.uniform(0.1, 0.9, (33, 2))
        candidates.append(pts)

    # Evaluate and pick best
    best_pts = None
    best_c = float('inf')
    
    for pts in candidates:
        c = _count_convex7_fast(pts)
        if c < best_c:
            best_c = c
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = candidates[0]
        best_c = _count_convex7_fast(best_pts)

    # Local search: Perturb points to break any remaining convex structures
    # Focus on moving points slightly to make inner points more "interior" or break alignments
    while time_left() > 5:
        # Pick a random point and move it slightly
        idx = rng.integers(0, 33)
        new_pts = best_pts.copy()
        # Move towards center? Or random?
        # Moving inner points towards center makes them more likely to be interior.
        # Moving outer points outwards makes them more likely to be hull.
        # Let's just do random small steps.
        step = rng.normal(0, 0.005, 2)
        new_pts[idx] += step
        new_pts[idx] = np.clip(new_pts[idx], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts

    return best_pts

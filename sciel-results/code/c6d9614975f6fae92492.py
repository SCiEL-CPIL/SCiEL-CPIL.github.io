import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0


def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming matching the evaluator's logic.
    """
    n = len(points)
    # Lexicographic sort: x first, then y
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        # Calculate angles
        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # Precompute cross products for turns: cross(a, b, c) = (b-a) x (c-b)
        # cross_q[i, j, k] = cross(q[i], q[j], q[k])
        cross_q = np.zeros((m, m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                v1x = q[j, 0] - q[i, 0]
                v1y = q[j, 1] - q[i, 1]
                for k in range(m):
                    if j == k:
                        continue
                    v2x = q[k, 0] - q[j, 0]
                    v2y = q[k, 1] - q[j, 1]
                    cross_q[i, j, k] = v1x * v2y - v1y * v2x

        # cross_p[i, j] = cross(p, q[i], q[j])
        cross_p = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            v1x = q[i, 0] - p[0]
            v1y = q[i, 1] - p[1]
            for j in range(m):
                v2x = q[j, 0] - q[i, 0]
                v2y = q[j, 1] - q[i, 1]
                cross_p[i, j] = v1x * v2y - v1y * v2x

        # cross_close1[i, j] = cross(q[i], q[j], p)
        cross_c1 = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                v1x = q[j, 0] - q[i, 0]
                v1y = q[j, 1] - q[i, 1]
                v2x = p[0] - q[j, 0]
                v2y = p[1] - q[j, 1]
                cross_c1[i, j] = v1x * v2y - v1y * v2x

        # cross_close2[i, j] = cross(q[j], p, q[i])
        cross_c2 = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                v1x = p[0] - q[j, 0]
                v1y = p[1] - q[j, 1]
                v2x = q[i, 0] - p[0]
                v2y = q[i, 1] - p[1]
                cross_c2[i, j] = v1x * v2y - v1y * v2x

        # DP: dp[l, s, i, j] where chain is p, q[s], ..., q[i], q[j]
        dp = np.zeros((7, m, m, m), dtype=np.int64)

        # Init l=2
        for s in range(m):
            for j in range(s + 1, m):
                if cross_p[s, j] > 1e-9:
                    dp[2, s, s, j] = 1

        # Fill l=3..6
        for l in range(3, 7):
            for s in range(m):
                for i in range(s + 1, m):
                    for j in range(i + 1, m):
                        val = 0
                        for k in range(s, i):
                            if dp[l-1, s, k, i] > 0:
                                if cross_q[k, i, j] > 1e-9:
                                    val += dp[l-1, s, k, i]
                        dp[l, s, i, j] = val

        # Sum up l=6
        for s in range(m):
            for i in range(s + 1, m):
                for j in range(i + 1, m):
                    count = dp[6, s, i, j]
                    if count == 0:
                        continue
                    if cross_c1[i, j] > 1e-9 and cross_c2[s, j] > 1e-9:
                        total += count

    return total


def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    
    # Check distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False
        
    # Check collinearity
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

    # Structural change: Use a "layered" initial configuration.
    # Instead of a single circle or random points, we use multiple concentric
    # circles with different radii and angular offsets. This creates a structure
    # where points are grouped by "depth", which can reduce the number of
    # convex chains that wrap around the entire set.
    
    # Strategy: 33 points.
    # Layer 1: 11 points on a small circle (r=0.1)
    # Layer 2: 11 points on a medium circle (r=0.3)
    # Layer 3: 11 points on a large circle (r=0.45)
    # With random angular offsets to break symmetry.
    
    def generate_layered_config(seed_offset=0):
        rng_local = np.random.default_rng(42 + seed_offset)
        pts = np.zeros((33, 2))
        
        # Layer 1
        n1 = 11
        r1 = 0.15
        offset1 = rng_local.uniform(0, 2*np.pi)
        angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + offset1
        pts[:n1, 0] = 0.5 + r1 * np.cos(angles1)
        pts[:n1, 1] = 0.5 + r1 * np.sin(angles1)
        
        # Layer 2
        n2 = 11
        r2 = 0.3
        offset2 = rng_local.uniform(0, 2*np.pi)
        angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + offset2
        pts[n1:n1+n2, 0] = 0.5 + r2 * np.cos(angles2)
        pts[n1:n1+n2, 1] = 0.5 + r2 * np.sin(angles2)
        
        # Layer 3
        n3 = 11
        r3 = 0.45
        offset3 = rng_local.uniform(0, 2*np.pi)
        angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + offset3
        pts[n1+n2:, 0] = 0.5 + r3 * np.cos(angles3)
        pts[n1+n2:, 1] = 0.5 + r3 * np.sin(angles3)
        
        # Add small noise to avoid exact symmetries and collinearities
        pts += rng_local.normal(0, 0.01, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        
        return pts

    # Try a few different layered configurations to find a good starting point
    best_pts = None
    best_c = np.inf
    
    for seed in range(10):
        if time_left() < 60:
            break
        pts = generate_layered_config(seed)
        if _is_valid_fast(pts):
            c = _count_convex7_fast(pts)
            if c < best_c:
                best_c = c
                best_pts = pts

    if best_pts is None:
        # Fallback to random
        best_pts = rng.random((33, 2)) * 0.85 + 0.075
        best_pts += rng.normal(0, 0.005, (33, 2))
        best_pts = np.clip(best_pts, 0.01, 0.99)
        if not _is_valid_fast(best_pts):
            best_pts = np.linspace(0.1, 0.9, 33).reshape(11, 3)
            best_pts = np.column_stack([best_pts[:, 0], best_pts[:, 1] * 0.5 + 0.25])
            best_pts += rng.normal(0, 0.01, (33, 2))
            best_pts = np.clip(best_pts, 0.01, 0.99)
        best_c = _count_convex7_fast(best_pts)

    # Local search with simulated annealing-like behavior
    no_improve_count = 0
    max_no_improve = 2000
    temperature = 0.05
    
    while time_left() > 30 and no_improve_count < max_no_improve:
        improved = False
        
        # Adaptive move size
        move_scale = 0.02 + 0.03 * (1 - no_improve_count / max_no_improve)
        
        # Try moving 1-3 points
        for _ in range(200):
            if time_left() < 30:
                break
            n_move = rng.integers(1, 4)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                # Gaussian perturbation
                new_pts[i] += rng.normal(0, move_scale, 2)
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            
            if not _is_valid_fast(new_pts):
                continue
            
            c = _count_convex7_fast(new_pts)
            
            # Accept if better, or with some probability if worse (to escape local minima)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve_count = 0
                break
            else:
                # Simulated annealing acceptance
                delta = c - best_c
                if delta < 10:  # Only accept small worsenings
                    p_accept = np.exp(-delta / (temperature + 1e-9))
                    if rng.random() < p_accept:
                        best_pts = new_pts
                        best_c = c
                        no_improve_count = 0
                        improved = True
                        break
        
        if not improved:
            no_improve_count += 1
            temperature *= 0.995  # Cool down
            
            # Occasionally try a larger jump to escape deep local minima
            if no_improve_count % 100 == 0:
                n_move = rng.integers(5, 15)
                idx = rng.choice(33, size=n_move, replace=False)
                new_pts = best_pts.copy()
                for i in idx:
                    new_pts[i] = rng.random(2) * 0.85 + 0.075
                
                if _is_valid_fast(new_pts):
                    c = _count_convex7_fast(new_pts)
                    if c < best_c:
                        best_c = c
                        best_pts = new_pts
                        no_improve_count = 0
                        temperature = 0.05  # Reheat

    return best_pts

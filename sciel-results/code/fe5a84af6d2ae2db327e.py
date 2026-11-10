import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

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

def _count_convex7(points):
    """
    Count the number of 7-point subsets in convex position using the dynamic program
    described in the task contract.
    
    Points are sorted lexicographically. For each pivot i, later points are sorted by
    atan2 angle around it. A turn p->q->r counts as convex only if the cross product
    (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9, including closing turns back to the pivot.
    """
    n = len(points)
    # Sort lexicographically
    idx = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[idx]
    
    total = 0
    eps = 1e-9
    
    for i in range(n - 6):
        p = pts[i]
        later = pts[i + 1:]
        m = len(later)
        if m < 6:
            continue
        
        # Sort later points by angle around p
        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx2 = np.argsort(angles)
        q = later[idx2]
        
        # Precompute cross products for turns
        # cross[a, b, c] = (q[b]-q[a]) x (q[c]-q[b]) for a < b < c
        # We need this for the DP
        
        # dp[k][s][e] = number of convex chains of length k in q,
        # starting at index s, ending at index e, such that
        # p -> q[s] -> ... -> q[e] is a convex chain.
        
        # k=1: dp[1][s][s] = 1
        # k=2: dp[2][s][e] = 1 if s < e and turn(p, q[s], q[e]) > eps
        # k>2: dp[k][s][e] = sum_{m: s < m < e} dp[k-1][s][m] * I(turn(q[m], q[e]) > eps)
        #       where turn(q[m], q[e]) means the turn at q[m] when going to q[e]
        #       which is (q[m] - q[prev]) x (q[e] - q[m])
        #       But we don't know prev!
        
        # So the state must be dp[k][s][m] = number of chains of length k-1 ending at m,
        # starting at s. When extending to e, we need the turn at m, which depends on
        # the point before m. So we need to track the predecessor.
        
        # Let's use: dp[k][s][m] where the chain is p, q[s], ..., q[m]
        # and the point before m is implicitly tracked by the transition.
        
        # Actually, for the turn at m when extending to e, we need the point before m.
        # So we need a 4D state or we need to do it differently.
        
        # Alternative: dp[k][s][e] = sum over all valid chains of length k from s to e.
        # To compute dp[k][s][e], we sum over m < e:
        #   dp[k-1][s][m] * I(turn(prev, m, e) > eps)
        # But dp[k-1][s][m] aggregates over all possible prev.
        
        # So we need to keep the distribution of prev? No, that's not efficient.
        
        # Let's use the fact that for a fixed s and e, the chain is determined by
        # the sequence of points. We can do a DP where:
        # dp[k][m][e] = number of convex chains of length k ending at e, with the
        # second-to-last point being m. Then:
        # dp[k][m][e] = sum_{p < m} dp[k-1][p][m] * I(turn(p, m, e) > eps)
        
        # Base: dp[2][s][e] = 1 if turn(p, q[s], q[e]) > eps (chain: p, q[s], q[e])
        #       Here the "second-to-last" is q[s], last is q[e].
        
        # Then for k=3 to 6:
        # dp[k][m][e] = sum_{p < m} dp[k-1][p][m] * I(turn(q[p], q[m], q[e]) > eps)
        
        # After filling dp[6][m][e] for all m < e, we need to close the polygon:
        # Check turn(q[e], p, q[s]) and turn(p, q[s], ...) 
        # Wait, the closing turns are:
        #   ... -> q[e] -> p -> q[s] -> ...
        # So we need:
        #   turn(q[e], p, q[s]) > eps  and  turn(p, q[s], q[next]) > eps (already checked)
        # Actually, the chain is p -> q[s] -> ... -> q[e] -> p
        # The turns are:
        #   p -> q[s] -> q[next]: checked in dp
        #   ...
        #   q[prev] -> q[e] -> p: need to check
        #   q[e] -> p -> q[s]: need to check
        
        # So for each chain of length 6 in q (which gives 7 points total with p),
        # we need to check two additional turns:
        #   turn(q[e], p, q[s]) > eps
        #   turn(p, q[s], q[second]) > eps (this is already in the DP for k=2)
        # Wait, the DP already ensures all internal turns are convex.
        # The turns that are NOT checked by the DP are:
        #   The turn at p: from q[e] to p to q[s]
        #   The turn at q[e]: from q[prev] to q[e] to p
        
        # So we need to check:
        #   turn(q[prev], q[e], p) > eps
        #   turn(q[e], p, q[s]) > eps
        
        # But q[prev] is the point before q[e] in the chain, which is m in dp[6][m][e].
        
        # So for each (s, m, e) with dp[6][m][e] > 0:
        #   Check turn(q[m], q[e], p) > eps
        #   Check turn(q[e], p, q[s]) > eps
        #   If both, add dp[6][m][e] to total.
        
        # Let's implement this.
        
        # dp[k][m][e]: chains of length k in q, ending at e, with m as second-to-last.
        # k ranges from 2 to 6.
        # m < e, and m >= s (start index).
        
        # Actually, we also need to track the start index s.
        # So dp[k][s][m][e]? That's 4D. Size: 6 * 33 * 33 * 33 = 250k. OK.
        
        # But for k=2, s=m (the chain is p, q[s], q[e], so second-to-last is q[s]).
        # So dp[2][s][s][e] = 1 if turn(p, q[s], q[e]) > eps.
        
        # For k>2: dp[k][s][m][e] = sum_{p_idx < m} dp[k-1][s][p_idx][m] * I(turn(q[p_idx], q[m], q[e]) > eps)
        
        # This is O(m^4) per pivot, which is too slow for m=32 (32^4 = 1M, times 33 pivots = 33M).
        # In Python, this might be slow. Let's see if we can optimize.
        
        # For now, let's implement it and see.
        
        # Initialize
        # dp will be a list of 4D arrays, one for each k.
        # dp[k] has shape (m, m, m) where dp[k][s][m][e] is stored.
        # But we can use a 4D array of shape (7, m, m, m).
        
        dp = np.zeros((7, m, m, m), dtype=np.int64)
        
        # k=2
        for s in range(m):
            for e in range(s + 1, m):
                # turn(p, q[s], q[e])
                v1x = q[s, 0] - p[0]
                v1y = q[s, 1] - p[1]
                v2x = q[e, 0] - q[s, 0]
                v2y = q[e, 1] - q[s, 1]
                cross = v1x * v2y - v1y * v2x
                if cross > eps:
                    dp[2][s][s][e] = 1
        
        # k=3 to 6
        for k in range(3, 7):
            for s in range(m):
                for m_idx in range(s + 1, m):
                    for e in range(m_idx + 1, m):
                        # dp[k][s][m_idx][e] = sum_{p_idx < m_idx} dp[k-1][s][p_idx][m_idx] * I(turn(q[p_idx], q[m_idx], q[e]) > eps)
                        total_val = 0
                        for p_idx in range(s, m_idx):
                            prev_count = dp[k-1][s][p_idx][m_idx]
                            if prev_count == 0:
                                continue
                            # turn(q[p_idx], q[m_idx], q[e])
                            v1x = q[m_idx, 0] - q[p_idx, 0]
                            v1y = q[m_idx, 1] - q[p_idx, 1]
                            v2x = q[e, 0] - q[m_idx, 0]
                            v2y = q[e, 1] - q[m_idx, 1]
                            cross = v1x * v2y - v1y * v2x
                            if cross > eps:
                                total_val += prev_count
                        dp[k][s][m_idx][e] = total_val
        
        # Now, for each chain of length 6 in q (k=6), check the closing turns.
        for s in range(m):
            for m_idx in range(s + 1, m):
                for e in range(m_idx + 1, m):
                    count = dp[6][s][m_idx][e]
                    if count == 0:
                        continue
                    # Check turn(q[m_idx], q[e], p)
                    v1x = q[e, 0] - q[m_idx, 0]
                    v1y = q[e, 1] - q[m_idx, 1]
                    v2x = p[0] - q[e, 0]
                    v2y = p[1] - q[e, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= eps:
                        continue
                    # Check turn(q[e], p, q[s])
                    v1x = p[0] - q[e, 0]
                    v1y = p[1] - q[e, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= eps:
                        continue
                    total += count
    
    return total

def _make_grid_init(n, rng):
    """
    Create a structured grid initialization.
    """
    # Use a 6x6 grid, remove 3 points to get 33
    x_coords = np.linspace(0.1, 0.9, 6)
    y_coords = np.linspace(0.1, 0.9, 6)
    
    pts = []
    for x in x_coords:
        for y in y_coords:
            pts.append((x, y))
    
    pts = np.array(pts)
    
    # Remove 3 random points to get 33
    if len(pts) > n:
        indices_to_remove = rng.choice(len(pts), size=len(pts) - n, replace=False)
        mask = np.ones(len(pts), dtype=bool)
        mask[indices_to_remove] = False
        pts = pts[mask]
    
    # Add small perturbation to avoid collinearity
    pts += rng.normal(0, 0.002, (n, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    return pts

def _hill_climb(pts, c, rng, deadline, safe_margin):
    """
    Perform hill climbing to reduce the number of convex 7-gons.
    """
    best_pts = pts.copy()
    best_c = c
    
    def time_left():
        return deadline - safe_margin - (time.time() - start_time)
    
    while time_left() > 5:
        # Try moving 1-3 random points
        n_move = rng.integers(1, 4)
        indices = rng.choice(33, size=n_move, replace=False)
        
        new_pts = best_pts.copy()
        for idx in indices:
            new_pts[idx] = best_pts[idx] + rng.normal(0, 0.05, 2)
            new_pts[idx] = np.clip(new_pts[idx], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c_new = _count_convex7(new_pts)
            if c_new < best_c:
                best_c = c_new
                best_pts = new_pts
    
    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Initialize with grid structure
    best_pts = _make_grid_init(33, rng)
    best_c = _count_convex7(best_pts)
    
    # Hill climbing
    best_pts, best_c = _hill_climb(best_pts, best_c, rng, _DEADLINE, _SAFE_MARGIN)
    
    return best_pts

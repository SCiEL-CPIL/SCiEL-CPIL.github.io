import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming matching the evaluator's logic.
    Optimized with vectorized inner sums where possible.
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
        # We need cross_q[k, i, j] > 1e-9 for the DP transition.
        # cross_q[k, i, j] = (q[i]-q[k]) x (q[j]-q[i])
        
        # Vectorized calculation of cross_q for all k, i, j
        # q is (m, 2)
        # We want a 3D array cross_mask[k, i, j]
        
        # v1 = q[i] - q[k]
        # v2 = q[j] - q[i]
        # cross = v1[0]*v2[1] - v1[1]*v2[0]
        
        # This is still O(M^3) to compute explicitly.
        # Let's compute it on the fly in the DP or precompute a 3D array.
        # M <= 32. 32^3 = 32768. This is small.
        
        cross_q = np.zeros((m, m, m), dtype=np.float64)
        # Vectorized:
        # q_i = q[None, :, :] (1, m, 2)
        # q_k = q[:, None, :] (m, 1, 2)
        # v1 = q_i - q_k (m, m, 2) -> v1[k, i, :]
        
        # q_j = q[None, None, :] (1, 1, m, 2) ? No.
        # We need v2 = q[j] - q[i].
        # v2[i, j, :] = q[j] - q[i].
        
        # Let's use broadcasting carefully.
        # v1[k, i, :] = q[i] - q[k]
        q_k = q[:, np.newaxis, :] # (m, 1, 2)
        q_i = q[np.newaxis, :, :] # (1, m, 2)
        v1 = q_i - q_k # (m, m, 2) -> v1[k, i, :]
        
        # v2[i, j, :] = q[j] - q[i]
        q_i2 = q[:, np.newaxis, :] # (m, 1, 2) -> q[i]
        q_j = q[np.newaxis, :, :] # (1, m, 2) -> q[j]
        v2 = q_j - q_i2 # (m, m, 2) -> v2[i, j, :]
        
        # We need cross for (k, i, j).
        # v1 is indexed by (k, i). v2 is indexed by (i, j).
        # We need to align indices.
        # cross[k, i, j] = v1[k, i, 0]*v2[i, j, 1] - v1[k, i, 1]*v2[i, j, 0]
        
        # v1_expanded = v1[:, :, np.newaxis, :] # (m, m, 1, 2)
        # v2_expanded = v2[np.newaxis, :, :, :] # (1, m, m, 2)
        # This results in (m, m, m, 2) which is 32^4 = 1M elements. Acceptable.
        
        v1_exp = v1[:, :, np.newaxis, :]
        v2_exp = v2[np.newaxis, :, :, :]
        
        # cross_val[k, i, j]
        cross_val = v1_exp[..., 0] * v2_exp[..., 1] - v1_exp[..., 1] * v2_exp[..., 0]
        
        cross_mask = cross_val > 1e-9

        # cross_p[i, j] = cross(p, q[i], q[j])
        # v1 = q[i] - p
        # v2 = q[j] - q[i]
        v1_p = q - p[None, :] # (m, 2)
        # v2_p[i, j] = q[j] - q[i]
        v2_p = v2 # (m, m, 2) -> v2_p[i, j, :]
        
        cross_p_val = v1_p[:, 0, np.newaxis] * v2_p[..., 1] - v1_p[:, 1, np.newaxis] * v2_p[..., 0]
        # cross_p_val[i, j]
        cross_p_mask = cross_p_val > 1e-9

        # cross_c1[i, j] = cross(q[i], q[j], p)
        # v1 = q[j] - q[i]
        # v2 = p - q[j]
        v1_c1 = v2 # (m, m, 2) -> v1_c1[i, j, :]
        v2_c1 = p[None, None, :] - q[np.newaxis, np.newaxis, :] # (1, 1, m, 2) -> v2_c1[i, j, :] is p - q[j]
        # Wait, v2_c1 needs to be aligned with i, j.
        # v2_c1[i, j, :] = p - q[j]
        # p is (2,). q[j] is (2,).
        # p_b = p[None, None, :] # (1, 1, 2)
        # q_j_b = q[None, :, :] # (1, m, 2) -> q[j]
        # v2_c1 = p_b - q_j_b # (1, m, 2) -> v2_c1[0, j, :]
        # We need v2_c1[i, j, :] to be p - q[j].
        # So v2_c1 = (p[None, None, :] - q[None, :, :]) # (1, m, 2)
        # Broadcast to (m, m, 2) by adding axis 0.
        
        v2_c1_full = (p[None, None, :] - q[None, :, :]) # (1, m, 2)
        v2_c1_full = v2_c1_full[np.newaxis, :, :] # (1, m, 2) -> broadcast to (m, m, 2)
        # Actually, v1_c1 is (m, m, 2). v2_c1_full is (1, m, 2).
        # cross_c1[i, j] = v1_c1[i, j, 0]*v2_c1_full[0, j, 1] - ...
        # This is tricky with broadcasting. Let's just loop for closing turns, they are O(M^2).
        
        cross_c1 = np.zeros((m, m), dtype=np.float64)
        cross_c2 = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                # cross(q[i], q[j], p)
                v1x = q[j, 0] - q[i, 0]
                v1y = q[j, 1] - q[i, 1]
                v2x = p[0] - q[j, 0]
                v2y = p[1] - q[j, 1]
                cross_c1[i, j] = v1x * v2y - v1y * v2x
                
                # cross(q[j], p, q[i])
                v1x = p[0] - q[j, 0]
                v1y = p[1] - q[j, 1]
                v2x = q[i, 0] - p[0]
                v2y = q[i, 1] - p[1]
                cross_c2[i, j] = v1x * v2y - v1y * v2x

        cross_c1_mask = cross_c1 > 1e-9
        cross_c2_mask = cross_c2 > 1e-9

        # DP
        dp = np.zeros((7, m, m, m), dtype=np.int64)

        # Init l=2
        # dp[2, s, s, j] = 1 if cross_p_mask[s, j]
        # We can set this directly.
        for s in range(m):
            js = np.where(cross_p_mask[s, :])[0]
            js = js[js > s] # j > s
            if len(js) > 0:
                dp[2, s, s, js] = 1

        # Fill l=3..6
        for l in range(3, 7):
            # dp[l, s, i, j] = sum_{k < i} dp[l-1, s, k, i] * cross_mask[k, i, j]
            # We can iterate s, i, j.
            for s in range(m):
                for i in range(s + 1, m):
                    # dp_prev = dp[l-1, s, :i, i] -> shape (i-s,) ? No, k ranges s..i-1.
                    # dp[l-1, s, k, i] for k in s..i-1.
                    prev_vals = dp[l-1, s, s:i, i] # shape (i-s,)
                    if np.any(prev_vals == 0):
                        # Optimize: only consider non-zero
                        pass
                    
                    # masks = cross_mask[s:i, i, :] -> shape (i-s, m)
                    # We want sum over k for each j.
                    # result[j] = sum_{k} prev_vals[k] * cross_mask[k, i, j]
                    # This is a dot product of prev_vals with each column of cross_mask[s:i, i, :]
                    
                    masks = cross_mask[s:i, i, :] # (i-s, m)
                    # result = masks.T @ prev_vals
                    # But we only care about j > i.
                    res = np.dot(masks.T, prev_vals)
                    
                    # Set dp[l, s, i, j] = res[j] for j > i
                    js = np.arange(i + 1, m)
                    dp[l, s, i, js] = res[js]

        # Sum up l=6
        for s in range(m):
            for i in range(s + 1, m):
                for j in range(i + 1, m):
                    count = dp[6, s, i, j]
                    if count > 0:
                        if cross_c1_mask[i, j] and cross_c2_mask[s, j]:
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

    # Strategy: 3 tight linear clusters
    # This should minimize convex 7-gons significantly.
    
    def generate_clustered_linear(seed_offset=0):
        rng_local = np.random.default_rng(42 + seed_offset)
        pts = np.zeros((33, 2))
        
        # 3 clusters
        centers = [0.2, 0.5, 0.8]
        n_per_cluster = 11
        
        for c_idx, cx in enumerate(centers):
            start_idx = c_idx * n_per_cluster
            end_idx = start_idx + n_per_cluster
            
            # x coordinates: spread around cx
            # Use small spread to keep them linear-ish
            dxs = rng_local.uniform(-0.05, 0.05, n_per_cluster)
            # Sort to make it more linear? No, random order is fine, but sorted might be better for "line" feel
            dxs.sort()
            
            # y coordinates: very small noise around 0.5
            dys = rng_local.normal(0, 0.001, n_per_cluster)
            
            pts[start_idx:end_idx, 0] = cx + dxs
            pts[start_idx:end_idx, 1] = 0.5 + dys
            
        # Clip to bounds
        pts = np.clip(pts, 0.01, 0.99)
        
        # Ensure validity (no collinearity)
        # The noise should handle it.
        return pts

    # Generate initial config
    best_pts = generate_clustered_linear(0)
    
    if not _is_valid_fast(best_pts):
        # Add more noise if invalid
        best_pts += rng.normal(0, 0.01, (33, 2))
        best_pts = np.clip(best_pts, 0.01, 0.99)
        if not _is_valid_fast(best_pts):
             # Fallback random
             best_pts = rng.random((33, 2)) * 0.85 + 0.075
             best_pts += rng.normal(0, 0.005, (33, 2))
             best_pts = np.clip(best_pts, 0.01, 0.99)

    best_c = _count_convex7_fast(best_pts)
    
    # If C is already 0, we are done.
    if best_c == 0:
        return best_pts

    # Local search
    no_improve_count = 0
    max_no_improve = 500
    temperature = 0.05
    
    while time_left() > 30 and no_improve_count < max_no_improve:
        improved = False
        
        # Try moving 1 point
        for _ in range(100):
            if time_left() < 30:
                break
            
            idx = rng.integers(0, 33)
            new_pts = best_pts.copy()
            # Small perturbation
            new_pts[idx] += rng.normal(0, 0.01, 2)
            new_pts[idx] = np.clip(new_pts[idx], 0.01, 0.99)
            
            if not _is_valid_fast(new_pts):
                continue
            
            c = _count_convex7_fast(new_pts)
            
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve_count = 0
                break
            else:
                # Accept worse with low prob
                delta = c - best_c
                if delta < 5:
                    p_accept = np.exp(-delta / (temperature + 1e-9))
                    if rng.random() < p_accept:
                        best_pts = new_pts
                        best_c = c
                        no_improve_count = 0
                        improved = True
                        break
        
        if not improved:
            no_improve_count += 1
            temperature *= 0.99

    return best_pts

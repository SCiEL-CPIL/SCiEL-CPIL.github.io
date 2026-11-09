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

        # Precompute cross product matrix:
        # cross[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        # We need this for DP transitions.
        # cross_shape: (m, m, m) where cross[a,b,c] is defined for a<b<c
        # For efficiency, precompute turn(a, b, c) for all a < b < c
        # turn(a, b, c) = (q[b]-q[a]) x (q[c]-q[b])
        # = (q[b,0]-q[a,0])*(q[c,1]-q[b,1]) - (q[b,1]-q[a,1])*(q[c,0]-q[b,0])

        # Precompute turn matrix for all triples a < b < c
        # turn_mat[a, b, c] = cross product
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                # Vectorize over c
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        # Also precompute turn from pivot p to q[s] to q[c]
        # turn_p[s, c] = (q[s]-p) x (q[c]-q[s])
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        # dp[l][s][p_idx][c_idx] = number of convex chains of length l
        # starting at index s, with previous point at p_idx, ending at c_idx
        # l ranges from 2 to 6
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        dp_curr = np.zeros((m, m, m), dtype=np.int64)

        # Initialize dp[2]: chains of length 2
        # dp[2][s][s][c] = 1 if turn(p, q[s], q[c]) > 1e-9
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:] > 1e-9)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        # Fill dp[3..6]
        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    # For each c_idx > p_idx, sum over k in [s, p_idx-1]
                    # where dp_prev[s, k, p_idx] > 0 and turn(k, p_idx, c_idx) > 1e-9
                    # Vectorize over c_idx
                    if not np.any(dp_prev[s, s:p_idx, p_idx] > 0):
                        continue
                    # turn_mat[k, p_idx, c_idx] for k in [s, p_idx-1], c_idx in [p_idx+1, m-1]
                    # Shape: (p_idx-s, m-p_idx-1)
                    turn_slice = turn_mat[s:p_idx, p_idx, p_idx+1:]
                    # dp_prev[s, s:p_idx, p_idx] has shape (p_idx-s,)
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    # For each c_idx, sum dp_slice[k] * (turn_slice[k, c_idx] > 1e-9)
                    # turn_slice > 1e-9 gives boolean matrix (p_idx-s, m-p_idx-1)
                    mask = turn_slice > 1e-9
                    # dp_curr[s, p_idx, p_idx+1:] = sum over k of dp_slice[k] * mask[k, :]
                    contrib = (dp_slice[:, np.newaxis] * mask).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            dp_prev, dp_curr = dp_curr, dp_prev

        # Sum up results for l=6
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue

                    # Check closing 1: (q[p_idx], q[c_idx], p)
                    # turn(q[p_idx], q[c_idx], p) = (q[c_idx]-q[p_idx]) x (p-q[c_idx])
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= 1e-9:
                        continue

                    # Check closing 2: (q[c_idx], p, q[s])
                    # turn(q[c_idx], p, q[s]) = (p-q[c_idx]) x (q[s]-p)
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

def _generate_initial_points(rng):
    """
    Generate a set of points designed to minimize convex 7-gons.
    Strategy: Place points in very dense clusters near the 4 corners and
    mid-edges to minimize the number of points in convex position.
    """
    pts = []
    
    # 5 cluster centers: 4 corners + 1 center
    centers = [
        (0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95),
        (0.5, 0.5)
    ]
    
    # Distribute 33 points: 7 per corner (28) + 5 in center
    for i, (cx, cy) in enumerate(centers):
        if i < 4:
            n_pts = 7
        else:
            n_pts = 5
        
        for _ in range(n_pts):
            # Very tight cluster
            x = cx + rng.normal(0, 0.015)
            y = cy + rng.normal(0, 0.015)
            pts.append((x, y))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 200:
        # Perturb points slightly
        pts += rng.normal(0, 0.002, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _hill_climb(pts, rng, time_limit):
    """
    Perform hill climbing to minimize the number of convex 7-gons.
    Uses adaptive step sizes and multiple-point moves to escape local minima.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 300
    step_size = 0.06
    min_step = 0.005
    max_step = 0.3
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        # Try moving 1 point
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        
        # Random new position within a neighborhood
        offset = rng.normal(0, step_size, 2)
        new_pts[i] = best_pts[i] + offset
        new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                # Reduce step size on improvement
                step_size = max(min_step, step_size * 0.9)
            elif c == best_c and best_c > 1:
                # Accept neutral moves to explore
                if rng.random() < 0.3:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # If stuck, try a larger jump with multiple points
        if not improved and no_improve % 3 == 0:
            n_move = rng.integers(2, 6)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                offset = rng.normal(0, step_size * 2.0, 2)
                new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size = max(min_step, step_size * 0.9)
                elif c == best_c and best_c > 1:
                    if rng.random() < 0.25:
                        best_pts = new_pts
                        no_improve = 0
        
        # Occasionally increase step size to escape local minima
        if no_improve % 20 == 0:
            step_size = min(max_step, step_size * 1.5)
        
        # Fine-tuning phase: if step size is very small, do precise adjustments
        if step_size < 0.02 and no_improve > 50:
            # Try very fine adjustments
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            offset = rng.normal(0, 0.003, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                elif c == best_c and best_c > 1:
                    if rng.random() < 0.2:
                        best_pts = new_pts
                        no_improve = 0
    
    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Generate initial points using a structured approach
    best_pts = _generate_initial_points(rng)
    best_c = _count_convex7_fast(best_pts)
    
    print(f"Initial C: {best_c}")
    
    # Perform hill climbing with multiple restarts
    best_ever_pts = best_pts.copy()
    best_ever_c = best_c
    
    restart_count = 0
    while time_left() > 30:
        # Run hill climbing for a chunk of time
        chunk_time = min(35, time_left() - 30)
        if chunk_time < 8:
            break
        best_pts, best_c = _hill_climb(best_pts, rng, chunk_time)
        
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
        
        print(f"After hill climb: C={best_c}, best_ever={best_ever_c}, t={time.time()-start_time:.1f}")
        
        # If we've found C=1, we can stop early
        if best_ever_c <= 1:
            break
        
        # Restart with a fresh initial configuration
        restart_count += 1
        if time_left() > 50:
            new_pts = _generate_initial_points(rng)
            new_c = _count_convex7_fast(new_pts)
            if new_c < best_ever_c:
                best_pts = new_pts
                best_c = new_c
                best_ever_pts = new_pts.copy()
                best_ever_c = new_c
                print(f"Restart improved to C: {best_c}")
            else:
                # Continue from current best
                best_pts = best_ever_pts.copy()
                best_c = best_ever_c
    
    # Final polish: very fine-tuning with small step sizes
    if time_left() > 15:
        print("Starting final polish...")
        polish_pts = best_ever_pts.copy()
        polish_c = best_ever_c
        polish_start = time.time()
        
        for _ in range(500):
            if time.time() - polish_start > time_left() - 10:
                break
            
            i = rng.integers(0, 33)
            new_pts = polish_pts.copy()
            offset = rng.normal(0, 0.002, 2)
            new_pts[i] = polish_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < polish_c:
                    polish_c = c
                    polish_pts = new_pts
                    if polish_c < best_ever_c:
                        best_ever_c = polish_c
                        best_ever_pts = polish_pts.copy()
        
        print(f"After polish: C={polish_c}, best_ever={best_ever_c}")
    
    return best_ever_pts
# EVOLVE-BLOCK-END

import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 25.0

def _is_valid_fast(pts):
    """Check validity of point configuration."""
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    # Check pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    # Check collinearity
    n = 33
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                area = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
                if abs(area) < 1e-9:
                    return False
    return True

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming over convex chains.
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

        # Precompute turn matrix: turn_mat[a,b,c] = cross product of (q[a]->q[b]) and (q[b]->q[c])
        # This is (qx-px)(ry-qy) - (qy-py)(rx-qx) for the turn at q[b]
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        # Precompute turn from pivot p to q[s] to q[t]
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        # DP: dp[l][s][e] = number of convex chains of length l starting at q[s] and ending at q[e]
        # where s < e and all intermediate points have indices between s and e
        # A convex chain p -> q[s] -> ... -> q[e] means all turns are left turns (convex)
        
        # Base case: chains of length 2 (just p -> q[s] -> q[e])
        # dp[2][s][e] = 1 if turn_p[s, e] > 1e-9
        dp = np.zeros((7, m, m), dtype=np.int64)
        for s in range(m):
            valid_e = np.where(turn_p[s, s + 1:] > 1e-9)[0]
            if len(valid_e) > 0:
                dp[2, s, s + 1:][valid_e] = 1

        # Extend chains
        for l in range(3, 7):
            for s in range(m):
                for e in range(s + 2, m):
                    # Chain p -> q[s] -> ... -> q[e] of length l
                    # Previous end was some point q[j] where s < j < e
                    # We need turn at q[j] from previous to q[e] to be convex
                    # And the previous chain of length l-1 ended at q[j]
                    # The turn at q[j] is: (q[j]-prev) x (q[e]-q[j])
                    # But we need to track the full chain, so we use:
                    # dp[l][s][e] = sum over j in (s, e) of dp[l-1][s][j] * [turn at j from (chain) to e is convex]
                    # The turn at j: we need the vector from the point before j to j, and from j to e
                    # This is tricky. Let me think again.
                    
                    # Actually, the turn condition at each intermediate point q[j] (not the first or last)
                    # is that the cross product of (q[j] - q[j-1]) and (q[j+1] - q[j]) > 1e-9
                    # where j-1 and j+1 are consecutive in the chain.
                    
                    # Let me redefine: dp[l][s][e] = number of ways to form a convex chain of l points
                    # starting at q[s] and ending at q[e], using only points with indices in [s, e],
                    # such that all internal turns are convex (left turns).
                    
                    # For l=2: dp[2][s][e] = 1 if turn_p[s,e] > 1e-9 (the turn at p from q[s] to q[e])
                    # Wait, that's the turn at p, not at q[s] or q[e].
                    
                    # Let me re-read the problem. The convex chain is p -> q[s] -> q[j1] -> ... -> q[e]
                    # The turns are:
                    # 1. At p: from (q[s]-p) to (q[j1]-q[s])? No, the turn is at each vertex.
                    # Actually, for a polygon p, q[s], q[j1], ..., q[e], the turns are at each vertex.
                    # The turn at q[k] is the cross product of (q[k] - q[k-1]) and (q[k+1] - q[k]).
                    # For the first point q[s], the previous point is p, so the turn at q[s] is
                    # (q[s] - p) x (q[next] - q[s]).
                    # For the last point q[e], we need the closing turns back to p.
                    
                    # So for the chain p -> q[s] -> q[j1] -> ... -> q[e] -> p:
                    # Turn at q[s]: (q[s]-p) x (q[j1]-q[s]) > 1e-9
                    # Turn at q[j1]: (q[j1]-q[s]) x (q[j2]-q[j1]) > 1e-9
                    # ...
                    # Turn at q[e]: (q[e]-q[j_{l-2}]) x (p-q[e]) > 1e-9
                    # Turn at p: (p-q[e]) x (q[s]-p) > 1e-9
                    # And all intermediate turns.
                    
                    # So dp should track chains where all internal turns (at q[s], q[j1], ..., q[e]) 
                    # are convex, and then we check the two closing turns at p.
                    
                    # Let me redo:
                    # dp[l][s][e] = number of convex chains of l points q[s], q[j1], ..., q[e] 
                    # (so l points total, including s and e) such that all turns at q[s], q[j1], ..., q[j_{l-2}]
                    # are convex (left turns). The turn at q[e] is NOT checked here (it will be checked 
                    # when closing the polygon).
                    
                    # Actually, let's say dp[l][s][e] = number of chains q[s] -> ... -> q[e] with l points
                    # where all turns at the first l-1 points are convex. The turn at the last point q[e]
                    # is not checked.
                    
                    # For l=1: dp[1][s][s] = 1 (trivial)
                    # For l=2: dp[2][s][e] = 1 if turn at q[s] from p to q[e] is convex, i.e., 
                    #   (q[s]-p) x (q[e]-q[s]) > 1e-9
                    # For l>2: dp[l][s][e] = sum over j in (s, e) of dp[l-1][s][j] * [turn at q[j] from q[j-1] to q[e] is convex]
                    # But we don't track q[j-1]. We need to track the previous point.
                    
                    # Let me use a different DP: dp[l][a][b] = number of chains of l points ending with ... -> q[a] -> q[b]
                    # where all turns are convex.
                    
                    # dp[2][a][b] = 1 if (q[a]-p) x (q[b]-q[a]) > 1e-9, for a < b
                    # dp[l][a][b] = sum over c < a of dp[l-1][c][a] * [(q[a]-q[c]) x (q[b]-q[a]) > 1e-9]
                    
                    # Then for a 7-gon p, q[s], ..., q[e], we need:
                    # dp[6][prev][e] where the chain has 6 points q[s],...,q[e] and all turns at q[s],...,q[prev] are convex
                    # Wait, 7-gon has 7 points: p and 6 q's. So the chain has 6 q-points.
                    # dp[6][a][b] where a and b are the last two q-points.
                    # Then we check: turn at q[b] from q[a] to p, and turn at p from q[b] to q[s].
                    # But we don't track q[s] in dp[6][a][b].
                    
                    # This is getting complex. Let me just use the original approach but more carefully.
                    
                    pass  # This branch won't execute; see below for the actual implementation
        
        # Let me just implement it more directly with a simpler DP
        # dp[l][s][e] = number of convex chains p -> q[s] -> ... -> q[e] of length l (l points after p)
        # where all turns at q[s], ..., q[e] (except the last one q[e]) are convex
        # Actually, let's say all turns at q[s] through q[j_{l-1}] are convex, where j_{l-1} is the second-to-last point.
        
        # Simpler: just count directly for small l
        # For 7-gons, we need 6 points from q. Let's enumerate all 6-tuples in order and check.
        # But that's O(m^6) which is too slow for m=27.
        
        # Let me use the DP properly:
        # dp[l][i][j] = number of ways to choose l points from q[i], q[i+1], ..., q[j] 
        # forming a convex chain starting at q[i] and ending at q[j], 
        # with all internal turns convex.
        # dp[1][i][i] = 1
        # dp[2][i][j] = 1 if turn at q[i] (from p) to q[j] is convex, i.e., turn_p[i, j] > 1e-9
        # dp[l][i][j] = sum over k in (i, j) of dp[l-1][i][k] * [turn at q[k] from q[?] to q[j] is convex]
        # Problem: we need to know the point before q[k] in the chain.
        
        # Alternative: dp[l][i][j] = number of convex chains of l points from q[i] to q[j]
        # dp[2][i][j] = 1 if turn_p[i,j] > 1e-9
        # dp[l][i][j] = sum over k in (i,j) of dp[l-1][i][k] * [turn_mat[i][k][j] > 1e-9]
        # where turn_mat[i][k][j] is the cross product of (q[k]-q[i]) and (q[j]-q[k])
        # Wait, that's not right either. The turn at q[k] depends on the previous point, not just q[i].
        
        # I think the cleanest way is:
        # f[l][a][b] = number of convex chains of l points ending at q[a] -> q[b] (a < b)
        # f[2][a][b] = 1 if turn_p[a, b] > 1e-9
        # f[l][a][b] = sum over c < a of f[l-1][c][a] * [turn_mat[c][a][b] > 1e-9]
        
        # Then for a 7-gon with points p, q[s], ..., q[e] (6 q-points), we need:
        # f[6][a][b] where the chain is q[s] -> ... -> q[a] -> q[b] with 6 points
        # But we need to know s to check the closing turn at p.
        
        # This is getting too complex for the time. Let me just use a simpler but potentially slower approach:
        # For each 6-tuple of indices s < j1 < j2 < j3 < j4 < j5, check if the 7-gon is convex.
        # That's C(m,6) which for m=27 is 296010, and for each we do 7 turn checks.
        # Total: ~2 million operations per pivot, and 27 pivots = ~54 million. Should be OK in Python with numpy.
        
        # Actually, let me just implement the straightforward enumeration.
        from itertools import combinations
        
        for combo in combinations(range(m), 6):
            s, j1, j2, j3, j4, j5 = combo
            # Check all 7 turns
            # Turn at p: (q[s]-p) x (q[j5]-p) -- wait, no
            # The polygon is p, q[s], q[j1], q[j2], q[j3], q[j4], q[j5]
            # Turn at q[s]: (q[s]-p) x (q[j1]-q[s])
            # Turn at q[j1]: (q[j1]-q[s]) x (q[j2]-q[j1])
            # Turn at q[j2]: (q[j2]-q[j1]) x (q[j3]-q[j2])
            # Turn at q[j3]: (q[j3]-q[j2]) x (q[j4]-q[j3])
            # Turn at q[j4]: (q[j4]-q[j3]) x (q[j5]-q[j4])
            # Turn at q[j5]: (q[j5]-q[j4]) x (p-q[j5])
            # Turn at p: (p-q[j5]) x (q[s]-p)
            
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[j1, 0] - q[s, 0]
            v2y = q[j1, 1] - q[s, 1]
            if v1x * v2y - v1y * v2x <= 1e-9:
                continue
            
            v1x = q[j1, 0] - q[s, 0]
            v1y = q[j1, 1] - q[s, 1]
            v2x = q[j2, 0] - q[j1, 0]
            v2y = q[j2, 1] - q[j1, 1]
            if v1x * v2y - v1y * v2x <= 1e-9:
                continue
            
            v1x = q[j2, 0] - q[j1, 0]
            v1y = q[j2, 1] - q[j1, 1]
            v2x = q[j3, 0] - q[j2, 0]
            v2y = q[j3, 1] - q[j2, 1]
            if v1x * v2y - v1y * v2x <= 1e-9:
                continue
            
            v1x = q[j3, 0] - q[j2, 0]
            v1y = q[j3, 1] - q[j2, 1]
            v2x = q[j4, 0] - q[j3, 0]
            v2y = q[j4, 1] - q[j3, 1]
            if v1x * v2y - v1y * v2x <= 1e-9:
                continue
            
            v1x = q[j4, 0] - q[j3, 0]
            v1y = q[j4, 1] - q[j3, 1]
            v2x = q[j5, 0] - q[j4, 0]
            v2y = q[j5, 1] - q[j4, 1]
            if v1x * v2y - v1y * v2x <= 1e-9:
                continue
            
            v1x = q[j5, 0] - q[j4, 0]
            v1y = q[j5, 1] - q[j4, 1]
            v2x = p[0] - q[j5, 0]
            v2y = p[1] - q[j5, 1]
            if v1x * v2y - v1y * v2x <= 1e-9:
                continue
            
            v1x = p[0] - q[j5, 0]
            v1y = p[1] - q[j5, 1]
            v2x = q[s, 0] - p[0]
            v2y = q[s, 1] - p[1]
            if v1x * v2y - v1y * v2x <= 1e-9:
                continue
            
            total += 1

    return total

def _generate_grid_initial(rng):
    """
    Generate points on a sparse grid to maximize collinearity.
    Use a 6x6 grid and pick 33 points from it, with small perturbations.
    """
    n = 33
    # Create a 6x6 grid
    grid_size = 6
    margin = 0.1
    step = (1 - 2 * margin) / (grid_size - 1)
    grid_x = np.linspace(margin, 1 - margin, grid_size)
    grid_y = np.linspace(margin, 1 - margin, grid_size)
    
    # All grid positions
    all_positions = []
    for x in grid_x:
        for y in grid_y:
            all_positions.append((x, y))
    all_positions = np.array(all_positions)
    
    # We have 36 positions, pick 33
    idx = rng.choice(len(all_positions), size=n, replace=False)
    pts = all_positions[idx].copy()
    
    # Add small perturbations to break exact collinearity
    pts += rng.normal(0, 0.008, (n, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 500:
        n_fix = rng.integers(1, 4)
        fix_idx = rng.choice(n, size=n_fix, replace=False)
        pts[fix_idx] += rng.normal(0, 0.015, (n_fix, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _generate_cluster_lines_initial(rng):
    """
    Generate points clustered on a few lines to maximize collinearity.
    """
    n = 33
    # Use 3 lines, each with about 11 points
    pts_list = []
    
    # Line 1: horizontal near bottom
    n1 = 12
    x1 = np.linspace(0.1, 0.9, n1)
    y1 = np.full(n1, 0.15)
    pts_list.append(np.column_stack([x1, y1]))
    
    # Line 2: horizontal near middle  
    n2 = 11
    x2 = np.linspace(0.1, 0.9, n2)
    y2 = np.full(n2, 0.5)
    pts_list.append(np.column_stack([x2, y2]))
    
    # Line 3: horizontal near top
    n3 = 10
    x3 = np.linspace(0.1, 0.9, n3)
    y3 = np.full(n3, 0.85)
    pts_list.append(np.column_stack([x3, y3]))
    
    pts = np.vstack(pts_list)
    
    # Add small perturbations
    pts += rng.normal(0, 0.012, (n, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 500:
        n_fix = rng.integers(1, 4)
        fix_idx = rng.choice(n, size=n_fix, replace=False)
        pts[fix_idx] += rng.normal(0, 0.02, (n_fix, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _generate_diagonal_lines_initial(rng):
    """
    Generate points on diagonal lines.
    """
    n = 33
    pts_list = []
    
    # Diagonal line 1: y = x
    n1 = 12
    t1 = np.linspace(0.1, 0.9, n1)
    pts_list.append(np.column_stack([t1, t1]))
    
    # Diagonal line 2: y = 1 - x
    n2 = 11
    t2 = np.linspace(0.1, 0.9, n2)
    pts_list.append(np.column_stack([t2, 1 - t2]))
    
    # Diagonal line 3: y = 0.5 + 0.3*(x - 0.5)
    n3 = 10
    t3 = np.linspace(0.1, 0.9, n3)
    pts_list.append(np.column_stack([t3, 0.5 + 0.3 * (t3 - 0.5)]))
    
    pts = np.vstack(pts_list)
    
    pts += rng.normal(0, 0.012, (n, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 500:
        n_fix = rng.integers(1, 4)
        fix_idx = rng.choice(n, size=n_fix, replace=False)
        pts[fix_idx] += rng.normal(0, 0.02, (n_fix, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _hill_climb(pts, rng, time_limit):
    """
    Hill climbing with grid-aware moves.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 80
    step_size = 0.02
    
    # Precompute grid positions
    grid_size = 8
    margin = 0.08
    step = (1 - 2 * margin) / (grid_size - 1)
    grid_x = np.linspace(margin, 1 - margin, grid_size)
    grid_y = np.linspace(margin, 1 - margin, grid_size)
    grid_positions = np.array([[x, y] for x in grid_x for y in grid_y])
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        move_type = rng.random()
        
        if move_type < 0.3:
            # Move to a grid position
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            grid_idx = rng.integers(0, len(grid_positions))
            new_pts[i] = grid_positions[grid_idx] + rng.normal(0, 0.005, 2)
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        elif move_type < 0.6:
            # Single point perturbation
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        elif move_type < 0.8:
            # Move toward a line (horizontal)
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            # Snap y to one of a few levels
            levels = [0.15, 0.35, 0.5, 0.65, 0.85]
            target_y = levels[rng.integers(0, len(levels))]
            new_pts[i, 1] = best_pts[i, 1] * 0.3 + target_y * 0.7 + rng.normal(0, 0.01)
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        else:
            # Multi-point move
            n_move = rng.integers(2, 5)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                offset = rng.normal(0, step_size * 1.3, 2)
                new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                no_improve = 0
                step_size *= 0.97
            elif c == best_c and best_c > 1:
                if rng.random() < 0.2:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Periodic kick
        if no_improve > 0 and no_improve % 15 == 0:
            n_kick = rng.integers(3, 7)
            kick_idx = rng.choice(33, size=n_kick, replace=False)
            new_pts = best_pts.copy()
            for i in kick_idx:
                new_pts[i] += rng.normal(0, step_size * 2, 2)
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
        
        if no_improve % 30 == 0:
            step_size = min(0.08, step_size * 1.15)

    return best_pts, best_c

def _simulated_annealing(pts, rng, time_limit):
    """
    Simulated annealing.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    current_pts = best_pts.copy()
    current_c = best_c
    
    start_time = time.time()
    T = 0.3
    T_min = 0.001
    cooling = 0.996
    
    while time.time() - start_time < time_limit:
        n_move = rng.integers(1, 4)
        idx = rng.choice(33, size=n_move, replace=False)
        new_pts = current_pts.copy()
        step_size = 0.015 + 0.025 * T / 0.3
        for i in idx:
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = current_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if not _is_valid_fast(new_pts):
            T *= cooling
            continue
        
        c_new = _count_convex7_fast(new_pts)
        
        if c_new < current_c:
            current_pts = new_pts
            current_c = c_new
            if c_new < best_c:
                best_c = c_new
                best_pts = new_pts.copy()
        else:
            delta = c_new - current_c
            if delta > 0:
                accept_prob = np.exp(-delta / max(T, T_min))
                if rng.random() < accept_prob:
                    current_pts = new_pts
                    current_c = c_new
        
        T *= cooling
        if T < T_min:
            T = 0.08
            if rng.random() < 0.25:
                n_kick = rng.integers(3, 8)
                kick_idx = rng.choice(33, size=n_kick, replace=False)
                current_pts[kick_idx] += rng.normal(0, 0.04, (n_kick, 2))
                current_pts = np.clip(current_pts, 0.01, 0.99)
                if _is_valid_fast(current_pts):
                    current_c = _count_convex7_fast(current_pts)
                else:
                    current_pts = best_pts.copy()
                    current_c = best_c
    
    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)
    
    init_strategies = [
        _generate_grid_initial,
        _generate_cluster_lines_initial,
        _generate_diagonal_lines_initial,
    ]
    
    best_ever_pts = None
    best_ever_c = float('inf')
    
    for init_fn in init_strategies:
        if time_left() < 30:
            break
        pts = init_fn(rng)
        c = _count_convex7_fast(pts)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
    
    if best_ever_pts is None:
        best_ever_pts = _generate_grid_initial(rng)
        best_ever_c = _count_convex7_fast(best_ever_pts)
    
    # Phase 1: Hill climbing
    if time_left() > 50:
        chunk = min(100, time_left() - 50)
        pts, c = _hill_climb(best_ever_pts, rng, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
    
    # Phase 2: Simulated annealing
    if time_left() > 50:
        chunk = min(100, time_left() - 50)
        pts, c = _simulated_annealing(best_ever_pts, rng, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
    
    # Phase 3: More hill climbing
    if time_left() > 40:
        chunk = min(80, time_left() - 40)
        pts, c = _hill_climb(best_ever_pts, rng, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
    
    # Phase 4: Final SA
    if time_left() > 30:
        rng2 = np.random.default_rng(123)
        chunk = min(60, time_left() - 30)
        pts, c = _simulated_annealing(best_ever_pts, rng2, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
    
    # Phase 5: Final fine-tuning
    if time_left() > 20:
        chunk = min(30, time_left() - 20)
        pts, c = _hill_climb(best_ever_pts, rng, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
    
    return best_ever_pts

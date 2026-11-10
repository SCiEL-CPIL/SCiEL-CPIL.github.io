import time
import numpy as np

_DEADLINE = 625.0
_SAFE_MARGIN = 5.0

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
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if abs(cross) < 1e-9:
                    return False
    return True

def _count_convex7(points):
    """Count convex 7-gons using O(N^3) DP per pivot."""
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]
    eps = 1e-9
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

        # turn(a, b, c) = (q[b]-q[a]) x (q[c]-q[b])
        # We need turn for triples where a < b < c in angular order
        # Precompute turn_mat[a, b, c] for a < b < c
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                for c in range(b + 1, m):
                    v2x = q[c, 0] - q[b, 0]
                    v2y = q[c, 1] - q[b, 1]
                    turn_mat[a, b, c] = v1x * v2y - v1y * v2x

        # turn_p[s, e] = turn(p, q[s], q[e]) = (q[s]-p) x (q[e]-q[s])
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            for e in range(s + 1, m):
                v1x = q[s, 0] - p[0]
                v1y = q[s, 1] - p[1]
                v2x = q[e, 0] - q[s, 0]
                v2y = q[e, 1] - q[s, 1]
                turn_p[s, e] = v1x * v2y - v1y * v2x

        # turn_close[s, e] = turn(q[e], p, q[s]) = (p-q[e]) x (q[s]-p)
        turn_close = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            for e in range(s + 1, m):
                v1x = p[0] - q[e, 0]
                v1y = p[1] - q[e, 1]
                v2x = q[s, 0] - p[0]
                v2y = q[s, 1] - p[1]
                turn_close[s, e] = v1x * v2y - v1y * v2x

        # turn_to_p[prev, e] = turn(q[prev], q[e], p) = (q[e]-q[prev]) x (p-q[e])
        turn_to_p = np.zeros((m, m), dtype=np.float64)
        for prev in range(m):
            for e in range(prev + 1, m):
                v1x = q[e, 0] - q[prev, 0]
                v1y = q[e, 1] - q[prev, 1]
                v2x = p[0] - q[e, 0]
                v2y = p[1] - q[e, 1]
                turn_to_p[prev, e] = v1x * v2y - v1y * v2x

        # DP: dp[l][a][b] = number of convex chains of l points from p,
        # starting at q[a], ending at q[b], where a < b,
        # and all intermediate turns are convex.
        # l=2: chain p -> q[a] (no turn yet, just 2 points)
        # Actually, let's define it as:
        # dp[l][a][b] = number of chains p -> q[i1] -> ... -> q[il] where
        #   i1 = a, il = b, a < i2 < ... < b, all turns convex.
        # For l=2: p -> q[a] -> q[b], need turn(p, q[a], q[b]) > eps
        # For l>2: sum over prev < b of dp[l-1][a][prev] * [turn(q[prev], q[b], next) > eps]
        # Wait, the turn at q[prev] when going to q[b] is turn(q[prev-1], q[prev], q[b])?
        # No. Let me redefine more carefully.
        
        # A convex chain p -> q[i1] -> q[i2] -> ... -> q[il] means:
        # - i1 < i2 < ... < il (angular order)
        # - turn(p, q[i1], q[i2]) > eps
        # - turn(q[ik], q[ik+1], q[ik+2]) > eps for k = 1..l-2
        # - (for closing) turn(q[il-1], q[il], p) > eps and turn(q[il], p, q[i1]) > eps
        
        # dp[l][a][b]: number of chains of length l (l points from p, so l-1 q-points)
        #   p -> q[i1] -> ... -> q[i_{l-1}] where i1=a, i_{l-1}=b
        #   with all internal turns convex.
        # l=1: just p, dp[1] is trivial (1 way for each "empty" chain)
        # l=2: p -> q[a], dp[2][a][a] = 1 (no turn to check)
        # l=3: p -> q[a] -> q[b], need turn(p, q[a], q[b]) > eps, dp[3][a][b] = 1 if valid
        # l=4: p -> q[a] -> q[prev] -> q[b], need turn(q[prev], q[b]...) wait.
        #   Actually the turn at q[prev] is turn(q[a], q[prev], q[b]).
        #   dp[4][a][b] = sum_{prev: a<prev<b, turn(q[a], q[prev], q[b])>eps} dp[3][a][prev]
        #   But dp[3][a][prev] = 1 if turn(p, q[a], q[prev]) > eps.
        
        # Let me use: dp[l][a][b] where l = number of q-points in the chain.
        # l=1: just q[a], dp[1][a][a] = 1
        # l=2: q[a] -> q[b], need turn(p, q[a], q[b]) > eps
        # l=3: q[a] -> q[prev] -> q[b], need turn(q[prev]...) 
        #   Actually the full chain is p -> q[a] -> q[prev] -> q[b]
        #   Turns: turn(p, q[a], q[prev]) and turn(q[a], q[prev], q[b])
        #   dp[3][a][b] = sum_{prev: a<prev<b} [turn(p,q[a],q[prev])>eps] * [turn(q[a],q[prev],q[b])>eps]
        #   But this is getting complicated. Let me use the edge-based DP.
        
        # dp[l][i][j] = number of convex chains from p with l q-points,
        #   starting at q[i], ending at q[j], i < j.
        # The chain is p -> q[i] -> ... -> q[j].
        # l=1: just p->q[i], dp[1][i][i] = 1
        # l=2: p->q[i]->q[j], need turn(p, q[i], q[j]) > eps
        # l>=3: p->q[i]->...->q[k]->q[j], need turn(q[k], q[j]...) 
        #   The last turn before j is at some point. Let's say the previous point is q[k].
        #   dp[l][i][j] = sum_{k: i<k<j, turn(q[k-1]...)} 
        #   Hmm, I need to track the second-to-last point.
        
        # Let me use dp[l][i][j] = number of chains p->q[i]->...->q[j] with l q-points,
        # where the turn at each intermediate point is convex.
        # For l=2: only turn to check is turn(p, q[i], q[j]).
        # For l>2: dp[l][i][j] = sum_{k: i<k<j} dp[l-1][i][k] * [turn(q[k-1], q[k], q[j]) > eps]
        #   But I don't track q[k-1]. 
        #   
        # Standard approach: dp[l][i][j] where the chain ends with edge (i,j).
        # dp[2][i][j] = 1 if turn(p, q[i], q[j]) > eps
        # dp[l][i][j] = sum_{k: i<k<j, turn(q[k], q[i], q[j]) > eps} dp[l-1][k][i]
        #   Wait, that's not right either. The turn at q[i] when coming from q[k] and going to q[j]
        #   is turn(q[k], q[i], q[j]).
        #   So dp[l][i][j] = sum_{k: i<k<j} dp[l-1][k][i] * [turn(q[k], q[i], q[j]) > eps]
        #   But dp[l-1][k][i] means chain ending with edge (k,i), i.e., ...->q[k]->q[i].
        #   And we want to extend to q[j]: ...->q[k]->q[i]->q[j].
        #   The turn at q[i] is turn(q[k], q[i], q[j]).
        #   This works!
        
        # For a 7-gon: p, q[i1], q[i2], ..., q[i6]
        # Chain: p -> q[i1] -> ... -> q[i6]
        # dp[6][i5][i6] gives chains of 6 q-points ending with edge (i5, i6).
        # We need to close: turn(q[i6], p, q[i1]) > eps and turn(q[i5], q[i6], p) > eps.
        # But dp[6][i5][i6] doesn't track i1. 
        # 
        # I need dp[l][start][end] or track the start.
        # Let me use dp[l][a][b] = number of chains p->q[a]->...->q[b] with l q-points,
        # all turns convex, a < b.
        # dp[2][a][b] = 1 if turn(p, q[a], q[b]) > eps
        # dp[l][a][b] = sum_{k: a<k<b} dp[l-1][a][k] * [turn(q[k], q[b]...)]
        #   The turn at q[k] when the chain is ...->q[k]->q[b] is turn(q[prev], q[k], q[b]).
        #   But dp[l-1][a][k] doesn't tell us q[prev].
        #
        # OK let me just use the 3D state: dp[l][a][prev][b] is too much.
        # Let me use the fact that for counting, I can use:
        # dp[l][a][b] where I track both start and end, and the transition
        # uses the turn at the second-to-last point.
        #
        # Actually, the cleanest is:
        # dp[l][i][j] = # convex chains from p with exactly l points (including p),
        #   where the last two points are q[i], q[j] (i < j).
        # So the chain is p -> ... -> q[i] -> q[j].
        # l=2: p -> q[i] (no, that's 2 points total: p and q[i]).
        #   Let me say l = total points in chain including p.
        #   l=2: p, q[i]. dp[2][i][i] = 1.
        #   l=3: p, q[i], q[j]. dp[3][i][j] = 1 if turn(p, q[i], q[j]) > eps.
        #   l=4: p, ..., q[i], q[j]. dp[4][i][j] = sum_{k: i<k<j} dp[3][k][i] * [turn(q[k], q[i], q[j]) > eps]
        #     Wait, dp[3][k][i] means chain p, q[k], q[i] with turn(p,q[k],q[i])>eps.
        #     Extend to q[j]: turn at q[i] is turn(q[k], q[i], q[j]).
        #   This works!
        #
        # For 7-gon: 7 points total, so l=7.
        # dp[7][i][j] where the chain is p -> ... -> q[i] -> q[j].
        # To close the 7-gon: need turn(q[j], p, q[first]) > eps and turn(q[i], q[j], p) > eps.
        # But I don't track q[first] in dp[7][i][j].
        #
        # I need dp[l][first][i][j] or similar. Let me use:
        # dp[l][a][i][j] = # chains p->q[a]->...->q[i]->q[j] with l total points.
        # That's O(m^4) per pivot which is 33^4 ~ 1.2M, times 33 pivots ~ 40M. Might be ok.
        # But let me try a different approach: for each pair (a, j), count chains.
        
        # Let me just do it with dp[l][a][j] where a is the first q-point and j is the last.
        # dp[3][a][j] = 1 if turn(p, q[a], q[j]) > eps
        # dp[l][a][j] = sum_{k: a<k<j} dp[l-1][a][k] * [turn(q[k-1]...)]
        #   Still the same problem.
        
        # Let me just use the 4D DP for correctness:
        # dp[l][a][i][j] where l = number of q-points, a = first, i = second-to-last, j = last.
        # l=2: dp[2][a][a][j] = 1 if turn(p, q[a], q[j]) > eps (chain p, q[a], q[j])
        #   Wait, that's 3 points: p, q[a], q[j]. l=2 q-points.
        # l=3: dp[3][a][i][j] = dp[2][a][a][i] * [turn(q[a], q[i], q[j]) > eps] for a<i<j
        #   dp[2][a][a][i] = 1 if turn(p, q[a], q[i]) > eps
        # l=4: dp[4][a][i][j] = sum_{k: a<k<i} dp[3][a][k][i] * [turn(q[k], q[i], q[j]) > eps]
        #   Wait, dp[3][a][k][i] means chain p, q[a], q[k], q[i].
        #   Extend to q[j]: turn at q[i] is turn(q[k], q[i], q[j]).
        #   But dp[3][a][k][i] has k as second-to-last, i as last.
        #   So dp[4][a][i][j] = sum_{k: a<k<i} dp[3][a][k][i] * [turn(q[k], q[i], q[j]) > eps]
        #   This works!
        #
        # For 7-gon: 7 points total = 1 pivot + 6 q-points.
        # dp[6][a][i][j] where a is first q-point, i is second-to-last, j is last.
        # Close: turn(q[j], p, q[a]) > eps and turn(q[i], q[j], p) > eps.
        # Sum over all a < i < j.
        
        # dp[l][a][i][j] for l q-points, l >= 2.
        # l=2: dp[2][a][a][j] = 1 if turn(p, q[a], q[j]) > eps
        # l>=3: dp[l][a][i][j] = sum_{k: a<k<i} dp[l-1][a][k][i] * [turn(q[k], q[i], q[j]) > eps]
        
        # Memory: m^4 = 33^4 ~ 1.2M per pivot, 33 pivots ~ 40M. OK.
        # Time: O(m^4 * m) per pivot? No, the sum is over k, so O(m^3) per (a,i,j) triple... 
        # Actually for each l, for each (a,i,j), sum over k: O(m). Total per l: O(m^4).
        # 5 layers (l=3..6): O(5*m^4) per pivot. m~30: 5*810000 = 4M per pivot. 33 pivots: 132M. OK.
        
        # Let me implement this.
        max_l = 6  # number of q-points
        # dp[a][i][j] for current l
        # Initialize for l=2
        dp2 = np.zeros((m, m, m), dtype=np.int64)
        for a in range(m):
            for j in range(a + 1, m):
                if turn_p[a, j] > eps:
                    dp2[a, a, j] = 1
        
        dp_prev = dp2
        for l in range(3, max_l + 1):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for a in range(m):
                for i in range(a + 1, m):
                    for j in range(i + 1, m):
                        # Sum over k: a < k < i
                        total_count = 0
                        for k in range(a + 1, i):
                            if dp_prev[a, k, i] > 0 and turn_mat[k, i, j] > eps:
                                total_count += dp_prev[a, k, i]
                        if total_count > 0:
                            dp_curr[a, i, j] = total_count
            dp_prev = dp_curr
        
        # Close the 7-gon
        for a in range(m):
            for i in range(a + 1, m):
                for j in range(i + 1, m):
                    if dp_prev[a, i, j] == 0:
                        continue
                    if turn_to_p[i, j] > eps and turn_close[a, j] > eps:
                        total += dp_prev[a, i, j]
    
    return total

def _make_cup_cap(n_cup, n_cap, rng):
    pts = []
    for i in range(n_cup):
        t = (i + 0.5) / n_cup
        x = 0.05 + 0.9 * t
        y = 0.05 + 0.15 * (t - 0.5) ** 2
        x += rng.normal(0, 0.002)
        y += rng.normal(0, 0.002)
        pts.append((x, y))
    for i in range(n_cap):
        t = (i + 0.5) / n_cap
        x = 0.05 + 0.9 * t
        y = 0.85 - 0.15 * (t - 0.5) ** 2
        x += rng.normal(0, 0.002)
        y += rng.normal(0, 0.002)
        pts.append((x, y))
    return np.array(pts)

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    best_pts = None
    best_c = float('inf')

    # Try multiple initial configurations
    splits = [(16, 17), (17, 16), (15, 18), (18, 15)]
    for n_cup, n_cap in splits:
        if n_cup + n_cap != 33:
            continue
        pts = _make_cup_cap(n_cup, n_cap, rng)
        pts = np.clip(pts, 0.01, 0.99)
        if not _is_valid_fast(pts):
            for _ in range(20):
                pts += rng.normal(0, 0.003, (33, 2))
                pts = np.clip(pts, 0.01, 0.99)
                if _is_valid_fast(pts):
                    break
        if not _is_valid_fast(pts):
            continue
        c = _count_convex7(pts)
        if c < best_c:
            best_c = c
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = rng.uniform(0.1, 0.9, (33, 2))
        best_c = _count_convex7(best_pts)

    # Simulated Annealing
    T = 0.05
    T_min = 1e-6
    alpha = 0.995

    while time_left() > 15:
        if T < T_min:
            T = T_min

        n_move = int(rng.integers(1, 4))
        indices = rng.choice(33, size=n_move, replace=False)
        new_pts = best_pts.copy()
        step_size = 0.03 * (T / 0.05) + 0.005
        for idx in indices:
            new_pts[idx] = best_pts[idx] + rng.normal(0, step_size, 2)
            new_pts[idx] = np.clip(new_pts[idx], 0.01, 0.99)

        if _is_valid_fast(new_pts):
            c = _count_convex7(new_pts)
            delta_c = c - best_c
            if delta_c < 0 or (delta_c > 0 and rng.random() < np.exp(-delta_c / max(T, 1e-10))):
                best_pts = new_pts
                best_c = c
        T *= alpha

    return best_pts

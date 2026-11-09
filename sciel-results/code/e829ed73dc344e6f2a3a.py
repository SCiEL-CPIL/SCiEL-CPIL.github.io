# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0


def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming over convex chains.
    Matches the evaluator's logic exactly.
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

        # Calculate angles around pivot p
        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # Precompute all cross products for turns
        # cross[k][i][j] = turn(q[k], q[i], q[j]) for k < i < j
        # But we also need turn(p, q[i], q[j]) for the first step.

        # dp[l][s][p_idx][c_idx] = number of convex chains of length l
        # starting at q[s], with previous point q[p_idx], ending at q[c_idx]
        # where s < p_idx < c_idx (indices in angle-sorted order)

        # For l=2: chain is p, q[s], q[c]
        # dp[2][s][s][c] = 1 if turn(p, q[s], q[c]) > 1e-9

        # For l>2: dp[l][s][p_idx][c_idx] = sum over k where s <= k < p_idx
        #   dp[l-1][s][k][p_idx] * I[turn(q[k], q[p_idx], q[c_idx]) > 1e-9]

        # For the final count (l=6): chain p, q[s], ..., q[p_idx], q[c_idx]
        # Check closing turns:
        #   turn(q[p_idx], q[c_idx], p) > 1e-9
        #   turn(q[c_idx], p, q[s]) > 1e-9

        # Use 3D array: dp[l][s][p_idx] where we iterate c_idx
        # Actually let's use dp as a dict or 4D array
        # m <= 32, so 4D array of size 7 x m x m x m is fine (~200k entries)

        # For efficiency, use a list of 3D arrays
        dp = [None] * 7

        # dp[2]: shape (m, m) - dp[2][s][c] = 1 if turn(p, q[s], q[c]) > 1e-9
        # Note: for l=2, p_idx = s, so we can use a 2D array
        dp[2] = np.zeros((m, m), dtype=np.int64)

        for s in range(m):
            for c in range(s + 1, m):
                v1x = q[s, 0] - p[0]
                v1y = q[s, 1] - p[1]
                v2x = q[c, 0] - q[s, 0]
                v2y = q[c, 1] - q[s, 1]
                cross = v1x * v2y - v1y * v2x
                if cross > 1e-9:
                    dp[2][s, c] = 1

        # For l=3..6: dp[l] has shape (m, m, m) - dp[l][s][p_idx][c_idx]
        for l in range(3, 7):
            dp[l] = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    for c_idx in range(p_idx + 1, m):
                        s_val = 0
                        for k in range(s, p_idx):
                            val = dp[l-1][s, k, p_idx] if l - 1 >= 3 else dp[2][s, k]
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
                        dp[l][s, p_idx, c_idx] = s_val

        # Sum up valid 7-gons from dp[6]
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp[6][s, p_idx, c_idx]
                    if count == 0:
                        continue

                    # Check closing turn 1: (q[p_idx], q[c_idx], p)
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= 1e-9:
                        continue

                    # Check closing turn 2: (q[c_idx], p, q[s])
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

    # Initialize with points clustered near corners of the unit square
    # This creates a dense, non-convex structure with few convex 7-gons
    corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    best_pts = np.zeros((33, 2))
    for i in range(33):
        corner = corners[i % 4]
        best_pts[i] = corner + rng.normal(0, 0.03, 2)
    best_pts = np.clip(best_pts, 0.01, 0.99)

    if not _is_valid_fast(best_pts):
        # Add more perturbation
        best_pts += rng.normal(0, 0.01, (33, 2))
        best_pts = np.clip(best_pts, 0.01, 0.99)

    best_c = _count_convex7_fast(best_pts)
    best_score = 1.0 / (1.0 + best_c)

    # Local search
    no_improve_count = 0
    max_no_improve = 800

    while time_left() > 30 and no_improve_count < max_no_improve:
        improved = False

        # Strategy 1: Move 1-2 points to random positions near corners
        for _ in range(40):
            if time_left() < 30:
                break
            n_move = rng.integers(1, 3)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                # Choose a random corner and place near it
                corner = corners[rng.integers(0, 4)]
                new_pts[i] = corner + rng.normal(0, 0.04, 2)
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)

            if not _is_valid_fast(new_pts):
                continue

            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve_count = 0
                break

        # Strategy 2: Small perturbations of single points
        if not improved:
            for _ in range(60):
                if time_left() < 30:
                    break
                i = rng.integers(0, 33)
                new_pts = best_pts.copy()
                new_pts[i] += rng.normal(0, 0.015, 2)
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)

                if not _is_valid_fast(new_pts):
                    continue

                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    improved = True
                    no_improve_count = 0
                    break

        # Strategy 3: Move a point to a completely random position
        if not improved:
            for _ in range(30):
                if time_left() < 30:
                    break
                i = rng.integers(0, 33)
                new_pts = best_pts.copy()
                new_pts[i] = rng.random(2) * 0.85 + 0.075

                if not _is_valid_fast(new_pts):
                    continue

                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    improved = True
                    no_improve_count = 0
                    break

        if not improved:
            no_improve_count += 1
            # Occasionally try a larger jump to escape local minima
            if no_improve_count % 150 == 0:
                n_move = rng.integers(4, 10)
                idx = rng.choice(33, size=n_move, replace=False)
                new_pts = best_pts.copy()
                for i in idx:
                    corner = corners[rng.integers(0, 4)]
                    new_pts[i] = corner + rng.normal(0, 0.05, 2)
                    new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)

                if _is_valid_fast(new_pts):
                    c = _count_convex7_fast(new_pts)
                    if c < best_c:
                        best_c = c
                        best_pts = new_pts
                        no_improve_count = 0

    return best_pts

# EVOLVE-BLOCK-END

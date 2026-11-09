# EVOLVE-BLOCK-START
import numpy as np


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 2^(k-2) + 1 = 33 points in
    general position (no three collinear) inside the unit square so as to MINIMIZE
    the number of convex 7-gons whose vertices are among the points.

    Strategy:
      1. Initialize points on a sparse, high-symmetry grid to maximize near-collinearity.
      2. Apply small deterministic jitter to break exact collinearity (general position).
      3. Use a stochastic hill-climbing search that moves points to nearby grid positions
         or small perturbations to reduce the count of convex 7-gons.

    Returns:
        points: np.ndarray of shape (33, 2) with x,y coordinates in [0,1]^2.
    """
    n = 33
    rng = np.random.RandomState(42)

    # --- Step 1: Initialize on a sparse grid ---
    # Use a grid of size ~6x6 = 36 points, pick 33 of them.
    # This gives many collinear/near-collinear triples.
    grid_size = 6
    margin = 0.1
    span = 1.0 - 2 * margin
    xs = np.linspace(margin, 1 - margin, grid_size)
    ys = np.linspace(margin, 1 - margin, grid_size)
    grid_points = np.array([[x, y] for x in xs for y in ys])  # 36 points

    # Pick 33 points, preferring those that create more collinearities
    # For simplicity, just take the first 33, but shuffle for variety
    idx = np.arange(grid_points.shape[0])
    rng.shuffle(idx)
    points = grid_points[idx[:n]].copy()

    # --- Step 2: Add small jitter to break collinearity ---
    # The collinearity check requires |cross| >= 1e-9, so we need to ensure
    # no three points are exactly collinear. Add tiny perturbations.
    epsilon = 1e-4
    points += rng.uniform(-epsilon, epsilon, size=points.shape)
    points = np.clip(points, 0.0, 1.0)

    # --- Step 3: Hill-climbing search to reduce convex 7-gon count ---
    # We'll use a simple approach: try moving individual points to nearby
    # grid positions or small perturbations, accepting if it reduces C.

    def count_convex_7gons(pts):
        """
        Count the number of 7-point subsets in convex position using the
        dynamic programming approach described in the task.
        """
        pts = np.asarray(pts, dtype=np.float64)
        n_pts = len(pts)
        if n_pts < 7:
            return 0

        # Sort lexicographically
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        pts_sorted = pts[order]

        C = 0
        # For each pivot point i, consider points j > i in sorted order
        # Sort later points by atan2 angle around pivot
        for i in range(n_pts - 6):
            pi = pts_sorted[i]
            # Points after i in sorted order
            later_idx = np.arange(i + 1, n_pts)
            if len(later_idx) < 6:
                break
            later_pts = pts_sorted[later_idx]

            # Compute angles from pi to each later point
            dx = later_pts[:, 0] - pi[0]
            dy = later_pts[:, 1] - pi[1]
            angles = np.arctan2(dy, dx)
            # Sort by angle
            angle_order = np.argsort(angles)
            later_sorted = later_pts[angle_order]
            later_sorted_idx = later_idx[angle_order]

            m = len(later_sorted)
            if m < 6:
                continue

            # DP: dp[j][k] = number of convex chains of length k ending at point j
            # among the later_sorted points, with pivot pi as the start
            # A chain pi -> p_{a1} -> p_{a2} -> ... -> p_{ak} is convex if all
            # turns are convex (left turns)

            # We want chains of length 6 (6 points after pivot = 7 total)
            # dp[j][1] = 1 for all j (chain of length 1: just pi -> p_j)
            # dp[j][k] = sum over i < j where turn pi->p_i->p_j is convex AND
            #            the chain pi->...->p_i of length k-1 is convex

            # Actually, the standard approach: 
            # A convex chain of length L from pivot means L points in angular order
            # such that each consecutive triple (including pivot) makes a left turn.

            # Let's define: for a fixed pivot and angular ordering of later points,
            # we count the number of subsequences of length 6 such that:
            # - The 6 points are in increasing angular order
            # - Each consecutive triple (pi, p_a, p_b) has a left turn
            # - The closing turn (p_last, pi) is also convex... wait, the task says
            #   "including the closing turns back to the pivot"

            # Re-reading: "a turn p -> q -> r counts as convex only if ... > 1e-9,
            # including the closing turns back to the pivot"
            # So for a 7-gon pi, p1, p2, ..., p6 (in angular order around pi),
            # we need:
            # - turn pi -> p1 -> p2 is convex
            # - turn p1 -> p2 -> p3 is convex
            # - ...
            # - turn p4 -> p5 -> p6 is convex
            # - turn p5 -> p6 -> pi is convex (closing)
            # - turn p6 -> pi -> p1 is convex (closing)
            # Wait, that's 7 turns for 7 points. Let me re-read.

            # "for each pivot the later points are sorted by atan2 angle around it,
            # and a turn p -> q -> r counts as convex only if ... > 1e-9, including
            # the closing turns back to the pivot"

            # I think the DP counts convex chains where the pivot is one vertex,
            # and we extend by adding points in angular order. The "closing turns"
            # suggest that when we have a complete polygon, we check the turns
            # that wrap around.

            # For simplicity and correctness, let's use a brute-force approach
            # for 7-gons since n=33 is small enough.
            # Number of 7-subsets from 33 is C(33,7) = 4,272,048 which is feasible.

        # Brute force: check all 7-point subsets
        from itertools import combinations
        C = 0
        for subset in combinations(range(n_pts), 7):
            pts_sub = pts[subset]
            # Check if these 7 points form a convex 7-gon
            # A set of points is in convex position iff they are the vertices
            # of their convex hull and all 7 points are on the hull.
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(pts_sub)
                if len(hull.vertices) == 7:
                    C += 1
            except Exception:
                pass
        return C

    def try_move(points, i, new_pos):
        """Try moving point i to new_pos, return new points if valid."""
        old = points[i].copy()
        points[i] = new_pos
        # Check validity
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            points[i] = old
            return None
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            points[i] = old
            return None
        # Check minimum distance
        min_sq_dist = np.inf
        for j in range(n):
            if j == i:
                continue
            d2 = np.sum((points[i] - points[j])**2)
            if d2 < min_sq_dist:
                min_sq_dist = d2
        if min_sq_dist < 1e-12:
            points[i] = old
            return None
        # Check collinearity (no three points collinear)
        for j in range(n):
            if j == i:
                continue
            for k in range(j+1, n):
                if k == i:
                    continue
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
                if abs(cross) < 1e-9:
                    points[i] = old
                    return None
        return points

    # Get initial count
    C_current = count_convex_7gons(points)

    # Hill climbing: try to reduce C
    max_iter = 500
    for iteration in range(max_iter):
        improved = False
        # Try moving each point
        for i in range(n):
            # Generate candidate moves
            candidates = []
            # Small perturbations
            for _ in range(10):
                delta = rng.uniform(-0.05, 0.05, size=2)
                new_pos = points[i] + delta
                new_pos = np.clip(new_pos, 0.0, 1.0)
                candidates.append(new_pos)
            # Grid positions nearby
            for gx in xs:
                for gy in ys:
                    new_pos = np.array([gx, gy])
                    if np.sum((new_pos - points[i])**2) > 1e-8:
                        candidates.append(new_pos)

            for new_pos in candidates:
                test_points = points.copy()
                result = try_move(test_points, i, new_pos)
                if result is None:
                    continue
                C_new = count_convex_7gons(result)
                if C_new < C_current:
                    points = result
                    C_current = C_new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try larger perturbations
            for i in range(n):
                for _ in range(5):
                    delta = rng.uniform(-0.15, 0.15, size=2)
                    new_pos = points[i] + delta
                    new_pos = np.clip(new_pos, 0.0, 1.0)
                    test_points = points.copy()
                    result = try_move(test_points, i, new_pos)
                    if result is None:
                        continue
                    C_new = count_convex_7gons(result)
                    if C_new < C_current:
                        points = result
                        C_current = C_new
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break

    # Final validation
    points = np.clip(points, 0.0, 1.0)
    assert points.shape == (33, 2)
    assert not np.any(np.isnan(points))
    assert not np.any(np.isinf(points))
    assert np.all(points >= -1e-6) and np.all(points <= 1 + 1e-6)

    # Check minimum distance
    for i in range(n):
        for j in range(i+1, n):
            d2 = np.sum((points[i] - points[j])**2)
            assert d2 >= 1e-12, f"Points {i},{j} too close: {d2}"

    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
                assert abs(cross) >= 1e-9, f"Collinear points {i},{j},{k}: {cross}"

    return points


# EVOLVE-BLOCK-END

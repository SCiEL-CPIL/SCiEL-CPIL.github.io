# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the minimum
    squared distance between any two points is >= the maximum squared norm of any point.
    Uses a lattice-based construction (E8 ⊕ D3) followed by greedy expansion.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    d = 11

    # ---------------------------------------------------------------
    # Step 1: Build E8 lattice points (240 points, all norm^2 = 4)
    # E8 = {x in Z^8 : sum(x_i) even} ∪ {x in (Z+1/2)^8 : sum(x_i) even}
    # The 240 minimal vectors have norm^2 = 4
    # ---------------------------------------------------------------
    def build_e8_minimal():
        """Return the 240 minimal vectors of E8 (norm^2 = 4)."""
        pts = []
        # Type 1: integer vectors with even sum, weight 4 (four ±1's, rest 0)
        # C(8,4) * 2^4 = 70 * 16 = 1120... that's too many.
        # Actually the 240 minimal vectors are:
        # - 112 of type (±1, ±1, 0,...,0) with even number of positions... no.
        # Let me recall: E8 minimal vectors (norm^2=2 in standard normalization, norm^2=4 in our):
        # Standard E8 minimal vectors have norm^2 = 2. There are 240 of them:
        #   112 of type (±1, ±1, 0, ..., 0): C(8,2)*2^2 = 28*4 = 112
        #   128 of type (±1/2, ±1/2, ..., ±1/2): 2^8 = 256, but only those with even sum of signs?
        #   Actually all 256 half-integer vectors have norm^2 = 8*(1/4) = 2.
        #   But 256 + 112 = 368 ≠ 240. Let me reconsider.
        #
        # E8 root system: 240 roots, each with norm^2 = 2.
        # They are:
        #   (±1, ±1, 0, 0, 0, 0, 0, 0): C(8,2) * 2^2 = 28 * 4 = 112
        #   (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs:
        #   2^7 = 128 (fix one sign, vary the other 7 with even parity)
        #   Total: 112 + 128 = 240. ✓
        #
        # To use integer coordinates, scale by 2:
        #   (±2, ±2, 0, ..., 0): norm^2 = 8, count = 112
        #   (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs: norm^2 = 8, count = 128
        #   Total: 240 points, all norm^2 = 8.
        #
        # Minimum distance between two E8 roots: the inner product of two distinct roots
        # can be -2, -1, 0, 1, 2. Distance^2 = 4 - 2*inner.
        # Min distance^2 = 4 - 2*2 = 0? No, that would mean same point.
        # Actually for distinct roots, inner product is in {-2, -1, 0, 1, 2} but
        # inner=2 means same root, inner=-2 means opposite root.
        # For distinct non-opposite roots: inner in {-1, 0, 1}, so dist^2 in {4, 2, 6}... wait
        # dist^2 = ||a||^2 + ||b||^2 - 2<a,b> = 2 + 2 - 2<a,b> = 4 - 2<a,b>
        # If <a,b> = 1: dist^2 = 2
        # If <a,b> = 0: dist^2 = 4
        # If <a,b> = -1: dist^2 = 6
        # If <a,b> = 2: same point (dist^2 = 0)
        # If <a,b> = -2: opposite (dist^2 = 8)
        # So min distance^2 between distinct E8 roots = 2.
        #
        # After scaling by 2: min distance^2 = 8, max norm^2 = 8. So 8 >= 8. ✓

        # Type 1: (±2, ±2, 0, ..., 0)
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)

        # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of -1's
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                pts.append(np.array(signs, dtype=np.int64))

        return pts

    e8_pts = build_e8_minimal()  # 240 points in R^8, all norm^2 = 8

    # ---------------------------------------------------------------
    # Step 2: Build D3 lattice minimal vectors (12 points, norm^2 = 2)
    # D3 = A3. The 12 minimal vectors of D3 (norm^2 = 2) are:
    # (±1, ±1, 0) and permutations: C(3,2)*2^2 = 3*4 = 12
    # After scaling by 2: (±2, ±2, 0) and permutations, norm^2 = 8, count = 12
    # Min distance^2 in D3 scaled: 8
    # ---------------------------------------------------------------
    def build_d3_minimal():
        pts = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        return pts

    d3_pts = build_d3_minimal()  # 12 points in R^3, all norm^2 = 8

    # ---------------------------------------------------------------
    # Step 3: Combine E8 ⊕ D3 to get 240 * 12 = 2880 points in R^11
    # All points have norm^2 = 8 + 8 = 16
    # Min distance^2: for two points (a,b) and (a',b') in E8⊕D3:
    #   dist^2 = ||a-a'||^2 + ||b-b'||^2
    #   If a ≠ a': ||a-a'||^2 >= 8 (min in E8 scaled)
    #   If b ≠ b': ||b-b'||^2 >= 8 (min in D3 scaled)
    #   If a = a' and b = b': same point
    #   So min dist^2 = 8 (when a=a' and b≠b', or a≠a' and b=b')
    #   Wait: if a=a' and b≠b', dist^2 = 0 + ||b-b'||^2 >= 8
    #   If a≠a' and b=b', dist^2 = ||a-a'||^2 + 0 >= 8
    #   If a≠a' and b≠b', dist^2 >= 8 + 8 = 16
    #   So min dist^2 = 8, max norm^2 = 16. 8 >= 16? NO! Invalid!
    #
    # I made an error. Let me reconsider.
    # We need min_dist^2 >= max_norm^2.
    # With E8⊕D3 scaled: max_norm^2 = 16, min_dist^2 = 8. 8 < 16. Invalid.
    #
    # I need to rescale differently. Let me use unscaled E8 and D3:
    # E8 roots: norm^2 = 2, min_dist^2 = 2
    # D3 roots: norm^2 = 2, min_dist^2 = 2
    # E8⊕D3: norm^2 = 4, min_dist^2 = 2 (when only one component differs)
    # 2 >= 4? No.
    #
    # The issue is that in a direct sum, the min distance is the min of the component distances,
    # but the norm adds. So we need a different approach.
    #
    # Let me think again. We need integer points in Z^11.
    # The insight from the cell findings: "Using direct sum of scaled E8 (240 pts) and D3 (12 pts)
    # lattices gives 252 points with uniform norm^2=8 and min dist^2=8"
    #
    # 252 = 240 + 12, not 240*12. So it's a union, not a product!
    # E8 points embedded in R^11 (padded with 0s in last 3 coords): 240 points, norm^2 = 8
    # D3 points embedded in R^11 (padded with 0s in first 8 coords): 12 points, norm^2 = 8
    # Union: 252 points, all norm^2 = 8.
    # Min distance: within E8 part: >= 8. Within D3 part: >= 8.
    # Between E8 point (a, 0) and D3 point (0, b): dist^2 = ||a||^2 + ||b||^2 = 8 + 8 = 16 >= 8. ✓
    # So min_dist^2 = 8, max_norm^2 = 8. Valid!
    # ---------------------------------------------------------------

    # Build the 252-point configuration
    base_points = []

    # E8 part: 240 points in first 8 coordinates, last 3 are 0
    for v8 in e8_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        base_points.append(v11)

    # D3 part: 12 points in last 3 coordinates, first 8 are 0
    for v3 in d3_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:] = v3
        base_points.append(v11)

    best_points = base_points[:]
    best_k = len(base_points)  # 252

    # ---------------------------------------------------------------
    # Step 4: Greedy expansion - try to add more integer points
    # Current state: max_norm^2 = 8, min_dist^2 = 8
    # To add point p: need ||p||^2 <= 8 and min_{v in S} ||p-v||^2 >= 8
    # ---------------------------------------------------------------

    # Generate all integer vectors in Z^11 with norm^2 <= 8
    # This is a finite set. Let's enumerate them.
    # Coordinates can be in range [-2, 2] (since 3^2 > 8)
    # Actually max coord is 2 (since 2^2=4, 3^2=9>8)

    def generate_candidates(max_norm_sq=8, d=11):
        """Generate all integer vectors in Z^d with 0 < ||v||^2 <= max_norm_sq."""
        candidates = []
        # Use recursion or itertools
        # Max coordinate magnitude: floor(sqrt(max_norm_sq)) = 2
        coord_range = list(range(-2, 3))  # -2, -1, 0, 1, 2

        # This is 5^11 = 48,828,125 which is too many to enumerate fully.
        # Let's be smarter: enumerate by weight (number of non-zero coords)
        # Weight 1: C(11,1) * 4 = 44 (±1, ±2 in one position)
        # Weight 2: C(11,2) * (4*4 + 4*1 + 1*4) ... this gets complex
        # Let's just do a bounded search with pruning

        # Actually, let's use a different approach: enumerate all vectors with norm^2 <= 8
        # by iterating over possible coordinate values with pruning.

        def gen(idx, current, norm_sq):
            if norm_sq > max_norm_sq:
                return
            if idx == d:
                if norm_sq > 0:  # exclude zero vector
                    candidates.append(current.copy())
                return
            for val in coord_range:
                new_norm = norm_sq + val * val
                if new_norm > max_norm_sq:
                    continue
                current[idx] = val
                gen(idx + 1, current, new_norm)
            current[idx] = 0

        gen(0, np.zeros(d, dtype=np.int64), 0)
        return candidates

    # This might be slow for d=11. Let's check the count.
    # Number of integer vectors in Z^11 with norm^2 <= 8:
    # This is the number of lattice points in a 11D ball of radius sqrt(8) ≈ 2.83
    # It's manageable but let's estimate: roughly 5^11 * (fraction in ball)
    # Actually let's just generate them efficiently.

    if time.time() < deadline - 120:
        candidates = generate_candidates(8, 11)

        # Remove candidates already in base_points
        base_set = set(tuple(p) for p in base_points)
        candidates = [p for p in candidates if tuple(p) not in base_set]

        # Sort candidates by norm^2 (prefer smaller norms for easier addition)
        candidates.sort(key=lambda p: np.sum(p ** 2))

        # Greedily add candidates
        current_set = base_points[:]
        for p in candidates:
            if time.time() > deadline - 60:
                break
            # Check minimum distance to all points in current_set
            # Use vectorized computation for speed
            diffs = np.array(current_set) - p[np.newaxis, :]
            dists_sq = np.sum(diffs ** 2, axis=1)
            min_dist = np.min(dists_sq)
            if min_dist >= 8:
                current_set.append(p)

        if len(current_set) > best_k:
            best_points = current_set
            best_k = len(current_set)

    # ---------------------------------------------------------------
    # Step 5: Try to add points with norm^2 = 9, 10, ... up to some limit
    # But this requires min_dist^2 >= new_max_norm, which is harder.
    # For now, stick with norm^2 <= 8.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Step 6: Try a different base - maybe use a larger norm configuration
    # If we use points with norm^2 = 12, we need min_dist^2 >= 12.
    # This might allow more points.
    # ---------------------------------------------------------------

    # Alternative: use the full E8 lattice (not just minimal vectors)
    # E8 has points with norm^2 = 4, 8, 12, ...
    # Points with norm^2 = 4 in E8: the 240 roots (already used)
    # Points with norm^2 = 8 in E8: more points
    # If we use E8 points with norm^2 = 8 (embedded in R^11), all have norm^2 = 8
    # and min distance within E8 is 2 (for roots) but for norm-8 points it could be larger.
    # This is getting complex. Let's stick with what we have.

    # ---------------------------------------------------------------
    # Step 7: Try to expand with norm^2 = 1, 2, 4, 5, 8 candidates
    # that might fit in gaps
    # ---------------------------------------------------------------

    if time.time() < deadline - 60:
        # Try adding norm-1 and norm-2 points
        for norm_sq_target in [1, 2, 4, 5]:
            if time.time() > deadline - 30:
                break
            # Generate vectors with exactly this norm^2
            for p in candidates:
                if np.sum(p ** 2) != norm_sq_target:
                    continue
                if time.time() > deadline - 30:
                    break
                if tuple(p) in set(tuple(q) for q in best_points):
                    continue
                diffs = np.array(best_points) - p[np.newaxis, :]
                dists_sq = np.sum(diffs ** 2, axis=1)
                min_dist = np.min(dists_sq)
                max_norm = max(np.sum(q ** 2) for q in best_points)
                if min_dist >= max(max_norm, norm_sq_target):
                    best_points.append(p)

    # ---------------------------------------------------------------
    # Final verification and return
    # ---------------------------------------------------------------
    points_arr = np.array(best_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)

    # Verify validity
    if len(points_arr) < 2 or points_arr.shape[1] != 11:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    # Check no zero vector
    zero_mask = np.all(points_arr == 0, axis=1)
    if np.any(zero_mask):
        points_arr = points_arr[~zero_mask]

    if len(points_arr) < 2:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    # Check the distance constraint
    k = len(points_arr)
    min_dist_sq = float('inf')
    max_norm_sq = 0

    for i in range(k):
        norm_sq = int(np.sum(points_arr[i] ** 2))
        if norm_sq > max_norm_sq:
            max_norm_sq = norm_sq
        for j in range(i + 1, k):
            diff = points_arr[i] - points_arr[j]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

    if min_dist_sq < max_norm_sq:
        # Invalid - fallback to the base 252-point configuration
        fallback = []
        for v8 in e8_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[:8] = v8
            fallback.append(v11)
        for v3 in d3_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[8:] = v3
            fallback.append(v11)
        points_arr = np.array(fallback, dtype=np.int64)

    return points_arr

# EVOLVE-BLOCK-END

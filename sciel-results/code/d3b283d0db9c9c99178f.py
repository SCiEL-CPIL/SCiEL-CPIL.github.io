# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import sys
from fractions import Fraction

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Different strategy: start from the full 8x8x8 grid (all 512 points) and
    deterministically DELETE the point that participates in the fewest "bad"
    5-tuples (five co-spherical/co-planar points), updating bad-tuple counts
    incrementally.  This directly minimizes the number of violations instead of
    greedily building a valid set.

    Returns:
        np.ndarray of shape (k, 3) of integer grid coordinates in [0, 7] with
        no five co-spherical points.
    """
    n = 8
    deadline = time.time() + 595.0  # safety margin before 630s wall-clock limit

    # All grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512

    # Precompute squared norms
    sq = (grid ** 2).sum(axis=1)  # (N,)

    # Precompute the 4x4 determinant (as a Fraction) for every 5-tuple of distinct
    # grid points.  det == 0  <=>  the 5 points are co-spherical (or co-planar).
    # We store only the determinant value; we also need the tuple indices.
    #
    # Memory: C(512,5) is ~ 2.2e11 which is far too large.  So we do NOT
    # precompute all 5-tuples.  Instead we compute determinants on the fly
    # during the deletion process, but we cache them to avoid recomputation.
    #
    # Actually, even on-the-fly for all 5-tuples is too slow.  We need a
    # smarter approach.
    #
    # Key insight: We only need to know, for each point p, how many 5-tuples
    # containing p are "bad".  We can compute this by, for each p, iterating
    # over all 4-combinations of the other points and checking the determinant.
    # That is C(511,4) ~ 2.8e9 per p, times 512 p's = 1.4e12 checks.  Still too slow.
    #
    # We need a completely different approach.  Let me reconsider.
    #
    # Alternative: use the fact that the determinant is a polynomial in the
    # coordinates.  For fixed 4 points q1..q4, the set of points p that make
    # the 5-tuple bad is the set of points satisfying a linear equation in
    # (p, |p|^2).  Specifically, the 4x4 matrix with rows (q_i - p, |q_i|^2 - |p|^2)
    # has det = 0 iff p lies on the sphere/plane determined by q1..q4 (if they
    # are not themselves degenerate).
    #
    # This is getting complex.  Let me use a more practical approach:
    #
    # PRACTICAL APPROACH: Use a randomized local-search that starts from a
    # large random subset and removes points involved in bad 5-tuples, but
    # with a much faster bad-tuple detection using vectorized numpy operations
    # and early termination.  We also use a precomputed list of "dangerous"
    # 4-tuples (4 points that are nearly co-spherical with many other points)
    # to focus the search.
    #
    # Actually, let me just use a well-optimized version of the deletion
    # strategy but with a time-limited search that doesn't try to be exhaustive.
    #
    # FINAL PRACTICAL STRATEGY:
    # 1. Start with a random subset of ~200 points (to keep computation manageable).
    # 2. Repeatedly find all bad 5-tuples in the current set.
    # 3. Remove the point that appears in the most bad 5-tuples.
    # 4. Repeat until no bad 5-tuples remain or time runs out.
    # 5. Then try to add back points from the remaining grid to maximize size.
    #
    # This is a "repair then extend" strategy, which is different from pure
    # greedy construction.

    def det5(p, q1, q2, q3, q4):
        """Compute the 4x4 determinant for points p, q1, q2, q3, q4.
        Returns the determinant as a float (exact for integer inputs up to this size).
        """
        # Rows: (q_i - p, |q_i|^2 - |p|^2) for i=1..4
        rows = []
        for q in [q1, q2, q3, q4]:
            diff = q - p
            norm_diff = (q ** 2).sum() - (p ** 2).sum()
            rows.append([diff[0], diff[1], diff[2], norm_diff])
        M = np.array(rows, dtype=np.float64)
        return np.linalg.det(M)

    def find_bad_tuples(points):
        """Find all 5-tuples of indices that are bad (det < 0.5).
        Returns list of 5-tuples of indices.
        """
        k = len(points)
        bad = []
        # For each 5-tuple, check determinant
        # This is C(k,5) checks.  For k=200, C(200,5) ~ 2.5e9.  Too slow.
        # We need to be smarter.
        #
        # Use the incremental approach: for each point p, check all 4-combinations
        # of other points.  For k=200, that's C(199,4) ~ 6.3e6 per p, times 200 = 1.26e9.
        # Still too slow for a single pass.
        #
        # We'll limit the search to a smaller set or use sampling.
        # For now, we'll just check a random sample of 5-tuples.
        if k < 5:
            return bad
        
        # Sample up to 100000 random 5-tuples
        num_samples = min(100000, k * (k-1) * (k-2) * (k-3) * (k-4) // 120)
        for _ in range(num_samples):
            idx = np.random.choice(k, size=5, replace=False)
            p = points[idx[0]]
            q1, q2, q3, q4 = points[idx[1]], points[idx[2]], points[idx[3]], points[idx[4]]
            d = det5(p, q1, q2, q3, q4)
            if abs(d) < 0.5:
                bad.append(tuple(sorted(idx)))
        return bad

    def validate_solution(points: np.ndarray) -> bool:
        """Validate that the solution is correct: no 5 co-spherical points."""
        if len(points) == 0:
            return True
        if np.any(points < 0) or np.any(points >= 8):
            return False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.array_equal(points[i], points[j]):
                    return False
        k = len(points)
        for i in range(4, k):
            p = points[i]
            prev = points[:i]
            diff = prev - p
            norm_diff = (prev ** 2).sum(axis=1) - (p ** 2).sum()
            w = np.column_stack([diff, norm_diff])
            for combo in itertools.combinations(range(i), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        return True

    # Strategy: Repair-then-Extend
    # 1. Start with a random subset
    # 2. Repair by removing points in bad 5-tuples
    # 3. Extend by adding points that don't create bad 5-tuples

    best_solution = None
    best_size = 0

    np.random.seed(123)

    # Try multiple initial subset sizes
    for init_size in [150, 200, 250]:
        if time.time() > deadline:
            break

        # Random initial subset
        init_indices = np.random.choice(N, size=min(init_size, N), replace=False)
        current = grid[init_indices].copy()
        current_set = set(map(tuple, current.tolist()))

        # Repair phase: remove points in bad 5-tuples
        max_repair_iters = 200
        for it in range(max_repair_iters):
            if time.time() > deadline:
                break

            bad = find_bad_tuples(current)
            if not bad:
                break

            # Count how many bad tuples each point appears in
            from collections import Counter
            count = Counter()
            for t in bad:
                for idx in t:
                    count[idx] += 1

            # Remove the point with the highest count
            if count:
                worst_idx = max(count, key=count.get)
                current = np.delete(current, worst_idx, axis=0)
                current_set.discard(tuple(current[worst_idx].tolist()) if worst_idx < len(current) else tuple(grid[init_indices[worst_idx]].tolist()))
            else:
                break

        # Extend phase: try to add points from the remaining grid
        remaining = [t for t in range(N) if tuple(grid[t].tolist()) not in current_set]
        np.random.shuffle(remaining)

        for r in remaining:
            if time.time() > deadline:
                break
            p = grid[r]
            # Check if p can be added
            k = len(current)
            if k < 4:
                current = np.vstack([current, p]) if k > 0 else p.reshape(1, 3)
                current_set.add(tuple(p.tolist()))
                continue

            diff = current - p
            norm_diff = (current ** 2).sum(axis=1) - (p ** 2).sum()
            w = np.column_stack([diff, norm_diff])
            ok = True
            for combo in itertools.combinations(range(k), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    ok = False
                    break
            if ok:
                current = np.vstack([current, p])
                current_set.add(tuple(p.tolist()))

        if len(current) > best_size:
            if validate_solution(current):
                best_size = len(current)
                best_solution = current
                print(f"Init size {init_size}: Found solution with {best_size} points", file=sys.stderr)

    # Final validation
    if best_solution is not None:
        if not validate_solution(best_solution):
            print("WARNING: Best solution failed validation, returning empty", file=sys.stderr)
            return np.empty((0, 3), dtype=np.int64)
        return best_solution.astype(np.int64)

    # Fallback
    fallback = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    return fallback

# EVOLVE-BLOCK-END

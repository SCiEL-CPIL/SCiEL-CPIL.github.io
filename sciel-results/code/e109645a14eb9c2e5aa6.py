# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in the 8x8x8 grid.
    Maximize the number of points from {0,...,7}^3 such that no 5 are co-spherical.
    """
    n = 8
    deadline = time.monotonic() + 600.0
    safety_margin = 10.0

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.int64)
    N = len(grid)

    # Map from point tuple to its squared norm (for O(1) lookup)
    point_to_sq = {}
    for i, p in enumerate(grid):
        point_to_sq[tuple(p)] = grid_sq[i]

    def get_p_sq(p):
        """Get squared norm of point p via dictionary lookup."""
        return point_to_sq[tuple(p)]

    def can_add(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: int) -> bool:
        """
        Check if p can be added without creating 5 co-spherical points.
        Uses integer arithmetic for exact determinant computation.
        """
        m = len(arr)
        if m < 4:
            return True

        # Build lifted vectors: w_i = (q_i - p, |q_i|^2 - |p|^2)
        # All components are integers, so determinants are integers.
        # A violation occurs iff |det| < 0.5, which for integers means det == 0.
        w = np.empty((m, 4), dtype=np.int64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq

        # For small m, check all C(m,4) combinations
        if m <= 12:
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                # Use exact integer determinant via fraction-free Gaussian elimination
                # For 4x4, compute det directly
                d = _det4_int(sub)
                if d == 0:
                    return False
            return True
        else:
            # For larger m, use batched approach with float det (safe since
            # integer det either 0 or |det|>=1, and float64 can represent
            # integers exactly up to 2^53)
            combos = list(itertools.combinations(range(m), 4))
            if len(combos) <= 10000:
                sub = w[np.array(combos)]  # (C, 4, 4)
                dets = np.linalg.det(sub.astype(np.float64))
                return not np.any(np.abs(dets) < 0.5)
            else:
                # Sequential with early termination
                for combo in combos:
                    sub = w[list(combo)]
                    d = _det4_int(sub)
                    if d == 0:
                        return False
                return True

    def _det4_int(m):
        """Compute determinant of 4x4 integer matrix exactly."""
        # Using the formula for 4x4 determinant
        a = m
        return (
            a[0,0]*(a[1,1]*(a[2,2]*a[3,3]-a[2,3]*a[3,2]) - a[1,2]*(a[2,1]*a[3,3]-a[2,3]*a[3,1]) + a[1,3]*(a[2,1]*a[3,2]-a[2,2]*a[3,1]))
            - a[0,1]*(a[1,0]*(a[2,2]*a[3,3]-a[2,3]*a[3,2]) - a[1,2]*(a[2,0]*a[3,3]-a[2,3]*a[3,0]) + a[1,3]*(a[2,0]*a[3,2]-a[2,2]*a[3,0]))
            + a[0,2]*(a[1,0]*(a[2,1]*a[3,3]-a[2,3]*a[3,1]) - a[1,1]*(a[2,0]*a[3,3]-a[2,3]*a[3,0]) + a[1,3]*(a[2,0]*a[3,1]-a[2,1]*a[3,0]))
            - a[0,3]*(a[1,0]*(a[2,1]*a[3,2]-a[2,2]*a[3,1]) - a[1,1]*(a[2,0]*a[3,2]-a[2,2]*a[3,0]) + a[1,2]*(a[2,0]*a[3,1]-a[2,1]*a[3,0]))
        )

    def greedy_pass(order: np.ndarray, deadline: float) -> tuple:
        """Run a greedy pass over the given order of points."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.int64)
        skipped = []

        for p in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            p_sq = get_p_sq(p)
            if can_add(arr, arr_sq, p, p_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([p_sq], dtype=np.int64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
            else:
                skipped.append(p)

        return arr, arr_sq, skipped

    def local_improve_multi(arr: np.ndarray, arr_sq: np.ndarray, candidates: list, deadline: float, max_passes: int = 50) -> tuple:
        """
        Multi-pass local improvement: repeatedly try to insert candidate points.
        Shuffles candidates each pass for diversity.
        """
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        remaining = list(candidates)
        rng_local = np.random.default_rng(12345)

        for _pass in range(max_passes):
            if time.monotonic() + safety_margin >= deadline:
                break
            if not remaining:
                break
            rng_local.shuffle(remaining)
            improved_this_pass = False
            for i, p in enumerate(remaining):
                if time.monotonic() + safety_margin >= deadline:
                    break
                p_sq = get_p_sq(p)
                if can_add(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
                    remaining.pop(i)
                    improved_this_pass = True
            if not improved_this_pass:
                break

        return arr, arr_sq, remaining

    def make_random_order(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return grid[rng.permutation(N)]

    def make_norm_sorted_order(reverse: bool = False) -> np.ndarray:
        idx = np.argsort(grid_sq)
        if reverse:
            idx = idx[::-1]
        return grid[idx]

    def make_corner_biased_order(seed: int) -> np.ndarray:
        """Bias toward corners and edges."""
        rng = np.random.default_rng(seed)
        corner_score = np.zeros(N)
        for i, p in enumerate(grid):
            for c in p:
                if c == 0 or c == 7:
                    corner_score[i] += 1
        noise = rng.random(N) * 0.5
        scores = corner_score + noise
        idx = np.argsort(scores)
        return grid[idx]

    def make_axis_sorted_order(axis: int, reverse: bool = False) -> np.ndarray:
        idx = np.argsort(grid[:, axis])
        if reverse:
            idx = idx[::-1]
        return grid[idx]

    def make_diagonal_order(seed: int) -> np.ndarray:
        """Sort by sum of coordinates (diagonal direction)."""
        rng = np.random.default_rng(seed)
        diag_scores = grid.sum(axis=1).astype(float) + rng.random(N) * 0.1
        idx = np.argsort(diag_scores)
        return grid[idx]

    best = None
    best_sq = None

    # Phase 1: Quick greedy passes with diverse orderings
    orderings = []
    for seed in range(8):
        orderings.append(('random', make_random_order(seed)))
    orderings.append(('norm_asc', make_norm_sorted_order(False)))
    orderings.append(('norm_desc', make_norm_sorted_order(True)))
    for seed in range(4):
        orderings.append(('corner', make_corner_biased_order(seed)))
    for axis in range(3):
        orderings.append((f'axis{axis}_asc', make_axis_sorted_order(axis, False)))
        orderings.append((f'axis{axis}_desc', make_axis_sorted_order(axis, True)))
    for seed in range(3):
        orderings.append(('diag', make_diagonal_order(seed)))

    for name, order in orderings:
        if time.monotonic() + safety_margin >= deadline:
            break
        arr, arr_sq, skipped = greedy_pass(order, deadline)
        # Light local improvement
        arr, arr_sq, skipped = local_improve_multi(arr, arr_sq, skipped, deadline, max_passes=5)

        if best is None or len(arr) > len(best):
            best = arr.copy()
            best_sq = arr_sq.copy()

    # Phase 2: Intensive local improvement on the best solution
    if best is not None and time.monotonic() + safety_margin < deadline:
        best_set = set(map(tuple, best))
        all_points = [p for p in grid if tuple(p) not in best_set]
        best, best_sq, _ = local_improve_multi(best, best_sq, all_points, deadline, max_passes=100)

    # Phase 3: Additional random restarts if time remains
    for i in range(40):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = make_random_order(100 + i)
        arr, arr_sq, skipped = greedy_pass(order, deadline)
        arr, arr_sq, skipped = local_improve_multi(arr, arr_sq, skipped, deadline, max_passes=5)

        if best is None or len(arr) > len(best):
            best = arr.copy()
            best_sq = arr_sq.copy()

    # Phase 4: Final intensive polish on best
    if best is not None and time.monotonic() + safety_margin < deadline:
        best_set = set(map(tuple, best))
        all_points = [p for p in grid if tuple(p) not in best_set]
        best, best_sq, _ = local_improve_multi(best, best_sq, all_points, deadline, max_passes=200)

    if best is None:
        best = np.empty((0, 3), dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END

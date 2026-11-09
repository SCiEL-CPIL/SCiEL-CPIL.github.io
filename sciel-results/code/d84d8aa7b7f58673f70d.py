# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside [-tol, 1+tol]^2
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
            return False
    # Non-overlap: for i < j, must not have |cx_i-cx_j| < t AND |cy_i-cy_j| < t
    # where t = (s_i + s_j)/2 - tol
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            if dx < t and dy < t:
                return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers


def _resolve_overlap(centers, sides, tol=1e-7):
    """
    Attempt to resolve overlaps by pushing squares apart.
    Returns modified centers, sides.
    """
    n = len(sides)
    for iteration in range(200):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    # Overlapping: push apart along the axis of smaller gap
                    if abs(dx) < abs(dy):
                        # Push horizontally
                        push = (t - abs(dx)) / 2.0 + tol
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        # Push vertically
                        push = (t - abs(dy)) / 2.0 + tol
                        if dy >= 0:
                            centers[j, 1] += push
                            centers[i, 1] -= push
                        else:
                            centers[j, 1] -= push
                            centers[i, 1] += push
                    moved = True
        if not moved:
            break
    return centers


def _try_grow_squares(centers, sides, tol=1e-7, max_iters=500):
    """
    Try to grow each square as much as possible while maintaining feasibility.
    Uses iterative bisection on each square's size.
    """
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            # Try to grow square i
            lo = sides[i]
            hi = 1.0  # maximum possible side
            # Binary search for max size
            for _ in range(30):
                mid = (lo + hi) / 2.0
                test_sides = sides.copy()
                test_sides[i] = mid
                # Check if square i fits inside unit square
                half = mid / 2.0
                if centers[i, 0] - half < -tol or centers[i, 0] + half > 1.0 + tol:
                    hi = mid
                    continue
                if centers[i, 1] - half < -tol or centers[i, 1] + half > 1.0 + tol:
                    hi = mid
                    continue
                # Check non-overlap with all others
                feasible = True
                for j in range(n):
                    if j == i:
                        continue
                    t = (mid + sides[j]) / 2.0 - tol
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    if dx < t and dy < t:
                        feasible = False
                        break
                if feasible:
                    lo = mid
                else:
                    hi = mid
            # If we can grow
            if lo > sides[i] + 1e-9:
                sides[i] = lo
                improved = True
        if not improved:
            break
    return sides


def _grow_square_max(centers, sides, i, tol=1e-7):
    """Find the maximum side length for square i given all other positions/sizes."""
    cx, cy = centers[i]
    # Bound from unit square
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0
    # Bound from each other square j:
    # For square i with center (cx,cy) side s and square j with center (cxj,cyj) side sj:
    # They overlap iff |cx-cxj| < (s+sj)/2 - tol AND |cy-cyj| < (s+sj)/2 - tol
    # To avoid overlap: |cx-cxj| >= (s+sj)/2 - tol OR |cy-cyj| >= (s+sj)/2 - tol
    # For each j, the constraint is: s <= 2*(dx + tol) - sj OR s <= 2*(dy + tol) - sj
    # where dx = |cx-cxj|, dy = |cy-cyj|
    # The binding constraint for j is: s <= max(2*(dx+tol)-sj, 2*(dy+tol)-sj)
    # But we need BOTH conditions to NOT hold simultaneously.
    # Actually: overlap iff dx < (s+sj)/2 - tol AND dy < (s+sj)/2 - tol
    # No overlap iff dx >= (s+sj)/2 - tol OR dy >= (s+sj)/2 - tol
    # i.e., s <= 2*dx + 2*tol - sj OR s <= 2*dy + 2*tol - sj
    # The max s that avoids overlap with j is: max(2*dx + 2*tol - sj, 2*dy + 2*tol - sj)
    # But we also need s > 0, so if both are negative, any s > 0 overlaps -> but that can't happen
    # if currently feasible.
    for j in range(len(sides)):
        if j == i:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        sj = sides[j]
        # Max s such that square i doesn't overlap j:
        # s <= 2*(dx + tol) - sj  OR  s <= 2*(dy + tol) - sj
        # The maximum s satisfying the OR is: max(2*(dx+tol)-sj, 2*(dy+tol)-sj)
        bound_j = max(2.0 * (dx + tol) - sj, 2.0 * (dy + tol) - sj)
        if bound_j < max_s:
            max_s = bound_j
    if max_s < sides[i] - 1e-12:
        return sides[i]  # Can't even maintain current size (shouldn't happen if valid)
    return max_s


def _try_push_to_corner(centers, sides, i, tol=1e-7):
    """
    Try to push square i toward each of the 4 corners and edges.
    Returns (new_centers, new_sides, improved) if a better packing is found.
    """
    n = len(sides)
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    half = sides[i] / 2.0
    
    # Candidate target positions: corners and edge midpoints
    targets = [
        (half, half),              # bottom-left corner
        (1.0 - half, half),        # bottom-right
        (half, 1.0 - half),        # top-left
        (1.0 - half, 1.0 - half),  # top-right
        (0.5, half),               # bottom edge mid
        (0.5, 1.0 - half),         # top edge mid
        (half, 0.5),               # left edge mid
        (1.0 - half, 0.5),         # right edge mid
    ]
    
    for tx, ty in targets:
        test_c = centers.copy()
        test_c[i, 0] = tx
        test_c[i, 1] = ty
        if not _is_valid(test_c, sides, tol):
            continue
        # Try growing all squares
        trial_s = sides.copy()
        for k in range(n):
            new_s = _grow_square_max(test_c, trial_s, k, tol)
            if new_s > trial_s[k] + 1e-9:
                trial_s[k] = new_s
        if _is_valid(test_c, trial_s, tol):
            obj = _objective(test_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = test_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _local_optimize(centers, sides, tol=1e-7, max_rounds=150):
    """
    Local optimization: alternate between growing squares and repositioning.
    Uses the fast analytic _grow_square_max.
    """
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    
    for rnd in range(max_rounds):
        # Step 1: Grow all squares in-place (iteratively since growing one may block another)
        for pass_num in range(3):
            new_sides = sides.copy()
            for i in range(n):
                new_s = _grow_square_max(centers, new_sides, i, tol)
                if new_s > new_sides[i] + 1e-9:
                    new_sides[i] = new_s
            if _is_valid(centers, new_sides, tol):
                sides = new_sides
                obj = _objective(centers, sides)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_sides = sides.copy()
                    best_centers = centers.copy()
            else:
                break
        
        # Step 2: Try pushing each square to corners/edges
        for i in range(n):
            new_c, new_s = _try_push_to_corner(centers, sides, i, tol)
            if _objective(new_c, new_s) > best_obj + 1e-9:
                best_obj = _objective(new_c, new_s)
                best_centers = new_c
                best_sides = new_s
                centers = new_c.copy()
                sides = new_s.copy()
        
        # Step 3: Random perturbations with growing/shrinking (simulated annealing style)
        rng = np.random.default_rng(seed=rnd * 1000 + 42)
        for i in range(n):
            for _ in range(5):
                # Try different perturbation sizes
                scale = 0.02 + 0.03 * rng.random()
                dx = rng.uniform(-scale, scale)
                dy = rng.uniform(-scale, scale)
                new_c = centers.copy()
                new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i]/2, 1.0 - sides[i]/2)
                new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i]/2, 1.0 - sides[i]/2)
                if not _is_valid(new_c, sides, tol):
                    continue
                trial_s = sides.copy()
                for k in range(n):
                    ns = _grow_square_max(new_c, trial_s, k, tol)
                    if ns > trial_s[k] + 1e-9:
                        trial_s[k] = ns
                if _is_valid(new_c, trial_s, tol):
                    obj = _objective(new_c, trial_s)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_centers = new_c.copy()
                        best_sides = trial_s.copy()
                        centers = new_c.copy()
                        sides = trial_s.copy()
        
        # Step 4: Try pairwise swaps of positions (helps escape local optima)
        if rnd % 3 == 0:
            for i in range(n):
                for j in range(i+1, n):
                    if abs(sides[i] - sides[j]) < 0.1:  # Only swap similar-sized squares
                        new_c = centers.copy()
                        new_c[i], new_c[j] = new_c[j].copy(), new_c[i].copy()
                        if _is_valid(new_c, sides, tol):
                            trial_s = sides.copy()
                            for k in range(n):
                                ns = _grow_square_max(new_c, trial_s, k, tol)
                                if ns > trial_s[k] + 1e-9:
                                    trial_s[k] = ns
                            if _is_valid(new_c, trial_s, tol):
                                obj = _objective(new_c, trial_s)
                                if obj > best_obj + 1e-9:
                                    best_obj = obj
                                    best_centers = new_c.copy()
                                    best_sides = trial_s.copy()
                                    centers = new_c.copy()
                                    sides = trial_s.copy()
        
        # Convergence check
        if rnd > 5 and abs(_objective(centers, sides) - best_obj) < 1e-12:
            can_improve = False
            for i in range(n):
                max_s = _grow_square_max(centers, sides, i, tol)
                if max_s > sides[i] + 1e-9:
                    can_improve = True
                    break
            if not can_improve:
                break
    
    return best_centers, best_sides


def _init_grid_packing():
    """Baseline 3x3 grid packing."""
    n = 10
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _init_heterogeneous_1():
    """
    One large square (side 0.5) in bottom-left, rest in top-right region.
    Large square: [0, 0.5] x [0, 0.5], center (0.25, 0.25)
    Remaining region: [0,1]x[0.5,1] and [0.5,1]x[0,0.5]
    Place 9 squares in the top strip [0,1]x[0.5,1] (height 0.5)
    and right strip [0.5,1]x[0,0.5] (width 0.5).
    Actually let's put all 9 in the top-right L-shape.
    Simpler: put 9 squares in the region above and to the right.
    Let's use a 3x3 grid in [0,1]x[0.5,1] for 6, and [0.5,1]x[0,0.5] for 3.
    Top strip: 3 cols x 2 rows = 6 squares, side = 0.5/3 = 1/6
    Right strip below: 3 cols x 1 row = 3 squares, side = 0.5/3 = 1/6
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    # Square 0: large
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    # Top strip [0,1]x[0.5,1]: 3 cols, 2 rows
    side_t = 1.0 / 3.0
    idx = 1
    for row in range(2):
        for col in range(3):
            centers[idx, 0] = (col + 0.5) * side_t
            centers[idx, 1] = 0.5 + (row + 0.5) * side_t
            sides[idx] = side_t
            idx += 1
    # Right strip [0.5,1]x[0,0.5]: 3 cols, 1 row
    side_r = 1.0 / 3.0
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_r
        centers[idx, 1] = 0.25
        sides[idx] = side_r
        idx += 1
    return centers, sides


def _init_heterogeneous_2():
    """
    Two large squares (side 0.4) and 8 small ones.
    Square 0: [0, 0.4] x [0, 0.4], center (0.2, 0.2)
    Square 1: [0.6, 1.0] x [0, 0.4], center (0.8, 0.2)
    Remaining: top strip [0,1]x[0.4,1] (height 0.6) and middle [0.4,0.6]x[0,0.4]
    8 squares in top strip: 4 cols x 2 rows, side = 0.6/4 = 0.15... 
    Let's do 4x2 in top: side = 1/4 = 0.25? No, height is 0.6, so 2 rows of 0.3.
    4 cols of 0.25, 2 rows of 0.3 -> but squares must be square.
    Use side = 0.25: 4 cols fit in width 1.0, 2 rows need height 0.5 <= 0.6. OK.
    That's 8 squares of side 0.25 in top strip.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    # 8 squares of side 0.25 in top strip [0,1]x[0.4, 0.9]
    side_s = 0.25
    idx = 2
    for row in range(2):
        for col in range(4):
            centers[idx, 0] = (col + 0.5) * side_s
            centers[idx, 1] = 0.4 + (row + 0.5) * side_s
            sides[idx] = side_s
            idx += 1
    return centers, sides


def _init_heterogeneous_3():
    """
    One very large square (side 0.5) and 9 small squares filling the rest.
    Large: [0,0.5]x[0,0.5]
    Rest: L-shape. Fill with 9 squares of side ~0.167 in a 3x3 grid in [0,1]x[0.5,1]
    plus [0.5,1]x[0,0.5].
    Actually same as heterogeneous_1 but let's try different sizes.
    9 squares in the L: top strip [0,1]x[0.5,1] has area 0.5, right strip [0.5,1]x[0,0.5] area 0.25.
    Total L area = 0.75. 9 squares, avg area = 0.75/9 = 0.0833, side ~ 0.289.
    But they must fit geometrically. Let's just use 1/6 side (0.167) for all 9.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    # Top strip [0,1]x[0.5,1]: 6 cols x 1 row? side=1/6, 6 cols = 1.0, 1 row height 1/6 <= 0.5
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    # Right strip [0.5,1]x[0,0.5]: 3 cols x 1 row? side=1/6, 3 cols = 0.5, fits in width 0.5
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_4():
    """
    1 large (0.5) + 2 medium (0.25) + 7 small, placed in feasible positions.
    Large: [0,0.5]x[0,0.5]
    Medium 1: [0,0.25]x[0.5,0.75] -> center (0.125, 0.625)
    Medium 2: [0.75,1.0]x[0.5,0.75] -> center (0.875, 0.625)
    Small squares fill remaining gaps.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.125, 0.625]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    # Remaining regions to fill with 7 small squares:
    # [0.25,0.75]x[0.5,0.75] (width 0.5, height 0.25)
    # [0,1]x[0.75,1.0] (width 1, height 0.25)
    # [0.5,1]x[0,0.5] (width 0.5, height 0.5)
    # Place small squares of side ~0.125
    side_s = 0.125
    idx = 3
    positions = [
        [0.375, 0.625], [0.625, 0.625],  # in middle gap
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],  # top strip
        [0.75, 0.25],  # right-bottom
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_5():
    """
    Two large squares (0.45) in opposite corners, 8 smaller filling the rest.
    Square 0: [0,0.45]x[0,0.45], center (0.225, 0.225)
    Square 1: [0.55,1.0]x[0.55,1.0], center (0.775, 0.775)
    8 squares of side ~0.15 in remaining L-shape + cross.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.225, 0.225]
    sides[0] = 0.45
    centers[1] = [0.775, 0.775]
    sides[1] = 0.45
    # Fill with 8 squares of side 0.15
    side_s = 0.15
    idx = 2
    positions = [
        [0.225, 0.575], [0.225, 0.875],  # left of top-right large
        [0.525, 0.225], [0.825, 0.225],  # below top-right large / right of bottom-left
        [0.525, 0.525],  # center
        [0.05, 0.575], [0.05, 0.875],    # far left top
        [0.825, 0.525],                  # right middle
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_6():
    """
    One large (0.5) bottom-left, one medium (0.3) top-left, rest fill.
    Large: [0,0.5]x[0,0.5]
    Medium: [0,0.3]x[0.7,1.0] -> center (0.15, 0.85)
    Fill remaining with 8 squares.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.15, 0.85]
    sides[1] = 0.3
    # Remaining regions:
    # [0.3,1]x[0.7,1] (width 0.7, height 0.3)
    # [0,1]x[0.5,0.7] (width 1, height 0.2)
    # [0.5,1]x[0,0.5] (width 0.5, height 0.5)
    side_s = 0.15
    idx = 2
    positions = [
        [0.45, 0.85], [0.65, 0.85], [0.85, 0.85],  # top right area
        [0.2, 0.6], [0.4, 0.6], [0.6, 0.6], [0.8, 0.6],  # middle strip
        [0.75, 0.25],  # bottom right
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_7():
    """
    Four medium squares (0.3) in corners, 6 small filling center.
    Corners: (0.15,0.15), (0.85,0.15), (0.15,0.85), (0.85,0.85)
    Center region: [0.3,0.7]x[0.3,0.7] -> 2x2 grid of side 0.2? That's 4 squares.
    Plus 2 more elsewhere.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    # 4 corner squares of side 0.3
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        centers[i] = [cx, cy]
        sides[i] = 0.3
    # Center 2x2 grid of side 0.2 in [0.3,0.7]x[0.3,0.7]
    idx = 4
    for row in range(2):
        for col in range(2):
            centers[idx, 0] = 0.3 + (col + 0.5) * 0.2
            centers[idx, 1] = 0.3 + (row + 0.5) * 0.2
            sides[idx] = 0.2
            idx += 1
    # 2 more squares in remaining gaps
    centers[idx] = [0.15, 0.5]
    sides[idx] = 0.2
    idx += 1
    centers[idx] = [0.85, 0.5]
    sides[idx] = 0.2
    idx += 1
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s safety margin
    
    # Generate multiple initial packings
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
        _init_heterogeneous_7(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for centers, sides in inits:
        if time.time() > deadline:
            break
        
        # First, resolve any overlaps
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _resolve_overlap(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to fix
            for _ in range(10):
                centers = _resolve_overlap(centers, sides)
                centers = _clamp_to_unit(centers, sides)
                if _is_valid(centers, sides):
                    break
        
        if not _is_valid(centers, sides):
            continue
        
        # Local optimization
        opt_centers, opt_sides = _local_optimize(centers, sides)
        
        # If time remaining, do a second pass with more rounds
        remaining = deadline - time.time()
        if remaining > 120:
            opt_centers2, opt_sides2 = _local_optimize(opt_centers, opt_sides, max_rounds=80)
            if _is_valid(opt_centers2, opt_sides2):
                obj2 = _objective(opt_centers2, opt_sides2)
                if obj2 > _objective(opt_centers, opt_sides):
                    opt_centers, opt_sides = opt_centers2, opt_sides2
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # Also try the original (pre-optimization) in case optimizer degraded
        orig_obj = _objective(centers, sides)
        if orig_obj > best_obj:
            best_obj = orig_obj
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # If no valid solution found, fall back to grid
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        # Final validation
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final polish: one more round of growth on the best solution
        final_sides = best_sides.copy()
        for i in range(10):
            ns = _grow_square_max(best_centers, final_sides, i, tol=1e-7)
            if ns > final_sides[i] + 1e-9:
                final_sides[i] = ns
        if _is_valid(best_centers, final_sides):
            if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                best_sides = final_sides
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END

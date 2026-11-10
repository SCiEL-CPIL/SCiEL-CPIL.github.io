# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients so first nonzero is positive and gcd divides all."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Exact incidence count via int64 arithmetic."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(lns.shape[0]):
        A, B, C = lns[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(vals == 0))
    return total

def _build_grid_points(r, c):
    """r x c grid of points."""
    xs = np.arange(1, c + 1, dtype=np.int64)
    ys = np.arange(1, r + 1, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _grid_lines(points, r, c):
    """Return the r row-lines and c column-lines for a grid, as (n,3) int64."""
    xs = np.arange(1, c + 1, dtype=np.int64)
    ys = np.arange(1, r + 1, dtype=np.int64)
    lines = []
    for y in ys:
        lines.append((0, 1, -int(y)))   # y = const
    for x in xs:
        lines.append((1, 0, -int(x)))   # x = const
    return np.array(lines, dtype=np.int64)

def _candidate_lines(points, max_lines=200):
    """All distinct lines through pairs of points, sorted by coverage desc, capped."""
    n = points.shape[0]
    counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm is not None:
                counts[norm] = counts.get(norm, 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    return [it[0] for it in items[:max_lines]]

def _line_point_matrix(points, lines):
    """Boolean matrix M[i, j] = point i on line j."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # M = (A*x + B*y + C == 0)
    A = lns[:, 0][:, None] * pts[None, :, 0]
    B = lns[:, 1][:, None] * pts[None, :, 1]
    C = lns[:, 2][:, None]
    return (A + B + C) == 0

def _select_lines(points, num_lines):
    """Greedy: pick lines by descending coverage, avoid duplicates."""
    cand = _candidate_lines(points)
    pts = np.asarray(points, dtype=np.int64)
    selected = []
    selected_set = set()
    for (A, B, C) in cand:
        if len(selected) >= num_lines:
            break
        if (A, B, C) in selected_set:
            continue
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        if int(np.sum(vals == 0)) == 0:
            continue
        selected.append((A, B, C))
        selected_set.add((A, B, C))
    # fill if short
    counter = 1
    while len(selected) < num_lines:
        cand_line = _normalize_line(1, 1, -counter)
        if cand_line not in selected_set:
            selected.append(cand_line)
            selected_set.add(cand_line)
        counter += 1
    return np.array(selected, dtype=np.int64)

def _swap_search(points, lines, deadline, max_iters=40000):
    """
    Local search: try swapping one point's x-coordinate with another's, or shifting a
    single coordinate, accepting any move that increases total incidences.
    Vectorized per candidate.
    """
    pts = points.astype(np.int64).copy()
    lns = lines.astype(np.int64)
    n = pts.shape[0]
    m = lns.shape[0]

    # Precompute per-line point incidence as a set of (x,y) for fast eval is heavy;
    # instead compute total via matrix each time we change a point (n*m ops, cheap).
    def total():
        M = _line_point_matrix(pts, lns)
        return int(np.sum(M))

    cur = total()
    # occupied set for uniqueness
    occ = set(map(tuple, pts.tolist()))

    def eval_move(new_pts):
        M = _line_point_matrix(new_pts, lns)
        return int(np.sum(M))

    start = time.time()
    improved = True
    while improved and (time.time() - start) < deadline - 15:
        improved = False
        # coordinate swap between two points
        for i in range(n):
            if (time.time() - start) > deadline - 15:
                break
            for j in range(i + 1, n):
                x_i, y_i = int(pts[i, 0]), int(pts[i, 1])
                x_j, y_j = int(pts[j, 0]), int(pts[j, 1])
                if x_i == x_j:
                    continue
                # candidate: swap x
                new_i = (x_j, y_i)
                new_j = (x_i, y_j)
                if new_i in occ and new_i != (x_i, y_i):
                    continue
                if new_j in occ and new_j != (x_j, y_j):
                    continue
                test = pts.copy()
                test[i, 0] = x_j
                test[j, 0] = x_i
                val = eval_move(test)
                if val > cur:
                    pts = test
                    occ.discard((x_i, y_i)); occ.discard((x_j, y_j))
                    occ.add(new_i); occ.add(new_j)
                    cur = val
                    improved = True
                    break
            if (time.time() - start) > deadline - 15:
                break
    return pts, cur

def _shift_search(points, lines, deadline):
    """Try shifting a single point by small (dx,dy) to increase incidences."""
    pts = points.astype(np.int64).copy()
    lns = lines.astype(np.int64)
    n = pts.shape[0]

    def total():
        M = _line_point_matrix(pts, lns)
        return int(np.sum(M))

    cur = total()
    occ = set(map(tuple, pts.tolist()))
    start = time.time()
    improved = True
    R = 3
    while improved and (time.time() - start) < deadline - 15:
        improved = False
        for i in range(n):
            if (time.time() - start) > deadline - 15:
                break
            x, y = int(pts[i, 0]), int(pts[i, 1])
            best_val = cur
            best_xy = (x, y)
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    cand = (nx, ny)
                    if cand in occ:
                        continue
                    test = pts.copy()
                    test[i, 0] = nx
                    test[i, 1] = ny
                    val = int(np.sum(_line_point_matrix(test, lns)))
                    if val > best_val:
                        best_val = val
                        best_xy = (nx, ny)
            if best_xy != (x, y):
                occ.discard((x, y))
                occ.add(best_xy)
                pts[i, 0] = best_xy[0]
                pts[i, 1] = best_xy[1]
                cur = best_val
                improved = True
    return pts, cur

def _validate(points, lines):
    """Return (points, lines) cleaned to satisfy all validity constraints."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)

    # dedup points
    seen = set()
    good = []
    for p in pts.tolist():
        t = (int(p[0]), int(p[1]))
        if t not in seen:
            seen.add(t)
            good.append(t)
    pts = np.array(good, dtype=np.int64)
    if pts.shape[0] < 54:
        # regenerate grid
        pts = _build_grid_points(6, 9)

    # dedup lines
    seen = set()
    good = []
    for l in lns.tolist():
        norm = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if norm is None:
            continue
        if norm not in seen:
            seen.add(norm)
            good.append(norm)
    lns = np.array(good, dtype=np.int64)
    if lns.shape[0] < 27:
        lns = _select_lines(pts, 27)

    return pts, lns

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 560  # safety margin vs 630s kill

    # 1. Strong deterministic starting point: 6x9 grid -> 108 incidences
    points = _build_grid_points(6, 9)
    lines = _grid_lines(points, 6, 9)
    cur = _count_incidences(points, lines)

    # 2. Optionally refine lines with greedy selection (should keep grid lines, which are optimal)
    if (time.time() - start_time) < deadline - 60:
        cand_lines = _select_lines(points, 27)
        val = _count_incidences(points, cand_lines)
        if val > cur:
            lines = cand_lines
            cur = val

    # 3. Local search to try to improve (swap then shift)
    if (time.time() - start_time) < deadline - 40:
        points, cur = _swap_search(points, lines, deadline)

    if (time.time() - start_time) < deadline - 25:
        points, cur = _shift_search(points, lines, deadline)

    # 4. Validate / clean
    points, lines = _validate(points, lines)

    return points, lines

# EVOLVE-BLOCK-END

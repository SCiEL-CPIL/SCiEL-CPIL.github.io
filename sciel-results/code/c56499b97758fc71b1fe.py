"""
Minimize ordinary lines (lines containing exactly 2 of the 12 points)
for 12 integer points with |x|,|y| <= 1e9.

Strategy:
1. Fast exact counting of ordinary lines via normalized line keys.
2. Multiple structured seeds (grids, pencils, star patterns, conic, etc.).
3. Constraint-satisfaction: generate candidate collinear triples from a
   lattice and use them as "attractors" in local search.
4. Aggressive local search with:
   - Single-point moves (various step sizes)
   - "Snap to candidate triple" moves (place a point to complete a triple)
   - Pair-swap moves
   - Simulated annealing to escape local minima
5. Final high-precision polish pass.
6. Time-bounded with explicit deadline guard.
"""

import time
import random
import math
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the points."""
    n = len(points)
    if n < 3:
        return 0
    line_count = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            g = math.gcd(abs(A), abs(B))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_count[key] = line_count.get(key, 0) + 1
    return sum(1 for c in line_count.values() if c == 1)


def _count_ordinary_lines_fast(points):
    """
    Faster count using per-point slope bucketing.
    For each point i, bucket other points by normalized direction (slope).
    A line with k points contributes C(k,2) pairs; it's ordinary iff k=2.
    We count pairs that lie on a line with no third point.
    """
    n = len(points)
    if n < 3:
        return 0
    # For each pair (i,j), check if there's a third point on the same line.
    # Use slope buckets per point.
    # A line is ordinary iff exactly 2 points are on it.
    # Total pairs = C(12,2) = 66.
    # For each pair, if no third point collinear, it's an ordinary line.
    # But multiple pairs on the same line: if 3 points on a line, that line
    # has C(3,2)=3 pairs but is NOT ordinary. So we need to count lines, not pairs.
    
    # Alternative: for each point i, compute slopes to all other points.
    # Points with same slope from i are collinear with i.
    # Build line groups.
    line_count = {}
    for i in range(n):
        x1, y1 = points[i]
        slopes = {}
        for j in range(n):
            if j == i:
                continue
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Normalize direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            key = (dx, dy)
            slopes[key] = slopes.get(key, 0) + 1
        # For each slope from point i, the line through i in that direction
        # has (count + 1) points total (including i).
        # But we need to avoid double-counting: a line with k points is counted
        # k times (once from each point on it). We want to count it once.
        # Instead, let's just use the pair-based method but deduplicate.
    # Fall back to the reliable method
    line_count = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            g = math.gcd(abs(A), abs(B))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_count[key] = line_count.get(key, 0) + 1
    return sum(1 for c in line_count.values() if c == 1)


def _are_all_collinear(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        x2, y2 = points[i]
        if (x2 - x0) * (y1 - y0) != (x1 - x0) * (y2 - y0):
            return False
    return True


def _valid(points):
    """Check distinctness, bounds, and not-all-collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        t = (p[0], p[1])
        if t in seen:
            return False
        seen.add(t)
    return not _are_all_collinear(points)


def _lines_with_2plus(points):
    """Return dict: line_key -> list of point indices on that line (only lines with >=2 points)."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            g = math.gcd(abs(A), abs(B))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    return {k: list(v) for k, v in line_pts.items()}


def _generate_collinear_triples(lattice_size=20):
    """
    Generate all collinear triples from a lattice of size lattice_size x lattice_size
    centered at origin. These serve as "attractor" patterns for the local search.
    Returns a list of (p1, p2, p3) tuples where each p is (x, y).
    """
    triples = []
    half = lattice_size // 2
    for x1 in range(-half, half + 1):
        for y1 in range(-half, half + 1):
            for x2 in range(-half, half + 1):
                for y2 in range(-half, half + 1):
                    if x1 == x2 and y1 == y2:
                        continue
                    dx = x2 - x1
                    dy = y2 - y1
                    # Check if there's a third lattice point on the line
                    # The line through (x1,y1) and (x2,y2) has direction (dx, dy).
                    # A third point (x3,y3) is on the line if (x3-x1)*dy == (y3-y1)*dx
                    # and it's a distinct lattice point.
                    # For efficiency, only consider primitive directions and check
                    # if the segment or extension contains another lattice point.
                    g = math.gcd(abs(dx), abs(dy))
                    if g >= 2:
                        # There's a midpoint or other point between
                        # Check midpoint
                        mx = x1 + dx // 2
                        my = y1 + dy // 2
                        if dx % 2 == 0 and dy % 2 == 0:
                            if -half <= mx <= half and -half <= my <= half:
                                triples.append(((x1, y1), (mx, my), (x2, y2)))
                    # Also check points beyond the segment
                    # Point at (x1 - dx, y1 - dy) if in range
                    x3 = x1 - dx
                    y3 = y1 - dy
                    if -half <= x3 <= half and -half <= y3 <= half:
                        triples.append(((x3, y3), (x1, y1), (x2, y2)))
                    # Point at (x2 + dx, y2 + dy) if in range
                    x3 = x2 + dx
                    y3 = y2 + dy
                    if -half <= x3 <= half and -half <= y3 <= half:
                        triples.append(((x1, y1), (x2, y2), (x3, y3)))
    # Deduplicate: sort each triple and use set
    unique = set()
    for t in triples:
        st = tuple(sorted(t))
        unique.add(st)
    return [list(t) for t in unique]


def _local_search(points, max_iters=8000, deadline=None, seed=None, triples=None):
    """Local search with simulated annealing to minimize ordinary lines."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    n = len(points)
    best = [list(p) for p in points]
    best_score = _count_ordinary_lines(best)
    cur = [list(p) for p in best]
    cur_score = best_score

    if deadline is not None and time.time() > deadline:
        return best, best_score

    # Simulated annealing parameters
    T = 2.0
    cooling = 0.9995
    min_T = 0.01

    # Precompute candidate triples for "snap to triple" moves
    if triples is None:
        triples = []

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if T < min_T:
            T = min_T

        move_type = rng.random()

        if move_type < 0.30:
            # --- Single-point move ---
            idx = rng.randint(0, n - 1)
            old_x, old_y = cur[idx]
            step = rng.choice([1, 1, 2, 2, 3, 5, 8, 13, 21, 50])
            new_x = old_x + rng.randint(-step, step)
            new_y = old_y + rng.randint(-step, step)

            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue

            new_points = [list(p) for p in cur]
            new_points[idx] = [new_x, new_y]

            if not _valid(new_points):
                continue

            score = _count_ordinary_lines(new_points)
            delta = score - cur_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur = new_points
                cur_score = score
                if score < best_score:
                    best = [list(p) for p in cur]
                    best_score = score

        elif move_type < 0.50:
            # --- Move a point onto an existing line (to create collinearity) ---
            idx = rng.randint(0, n - 1)
            # Find lines with >= 2 other points
            line_pts = _lines_with_2plus(cur)
            candidate_lines = []
            for key, pts_on_line in line_pts.items():
                if idx not in pts_on_line and len(pts_on_line) >= 2:
                    candidate_lines.append((key, pts_on_line))
            if not candidate_lines:
                continue
            key, pts_on_line = rng.choice(candidate_lines)
            # Get two points on the line to compute the line equation
            p1 = cur[pts_on_line[0]]
            p2 = cur[pts_on_line[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue
            # Place point on this line at a nearby integer position
            # Parametric: (p1x + t*dx, p1y + t*dy) for integer t
            # Try several t values
            best_t_score = None
            best_t = None
            old_x, old_y = cur[idx]
            for t in range(-20, 21):
                nx = p1[0] + t * dx
                ny = p1[1] + t * dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_points = [list(p) for p in cur]
                new_points[idx] = [nx, ny]
                if not _valid(new_points):
                    continue
                s = _count_ordinary_lines(new_points)
                if best_t_score is None or s < best_t_score:
                    best_t_score = s
                    best_t = t
            if best_t is not None:
                nx = p1[0] + best_t * dx
                ny = p1[1] + best_t * dy
                new_points = [list(p) for p in cur]
                new_points[idx] = [nx, ny]
                score = best_t_score
                delta = score - cur_score
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    cur = new_points
                    cur_score = score
                    if score < best_score:
                        best = [list(p) for p in cur]
                        best_score = score

        elif move_type < 0.65:
            # --- Snap to candidate triple: find a lattice triple where 2 points
            #     are already in cur (or can be matched by translation), and place
            #     the third point.
            if not triples:
                # Fall back to single-point move
                idx = rng.randint(0, n - 1)
                old_x, old_y = cur[idx]
                step = rng.choice([1, 2, 3, 5, 10])
                new_x = old_x + rng.randint(-step, step)
                new_y = old_y + rng.randint(-step, step)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                new_points = [list(p) for p in cur]
                new_points[idx] = [new_x, new_y]
                if not _valid(new_points):
                    continue
                score = _count_ordinary_lines(new_points)
                delta = score - cur_score
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    cur = new_points
                    cur_score = score
                    if score < best_score:
                        best = [list(p) for p in cur]
                        best_score = score
                continue

            # Pick a random triple
            tri = rng.choice(triples)
            # Try to match 2 of the 3 triple points to 2 existing cur points
            # via translation (and optionally reflection)
            found = False
            for perm_idx in range(3):
                # Try using tri[perm_idx] as the "new" point, and the other two
                # as anchors to existing points
                anchor1, anchor2 = tri[(perm_idx+1)%3], tri[(perm_idx+2)%3]
                new_pt = tri[perm_idx]
                # Find if there exist i != j in cur such that
                # cur[i] = anchor1 + offset and cur[j] = anchor2 + offset
                # for some offset (ox, oy)
                # i.e., cur[i] - anchor1 == cur[j] - anchor2
                # For each pair (i,j) in cur, check if (cur[i][0]-anchor1[0], cur[i][1]-anchor1[1])
                # equals (cur[j][0]-anchor2[0], cur[j][1]-anchor2[1])
                for i in range(n):
                    ox = cur[i][0] - anchor1[0]
                    oy = cur[i][1] - anchor1[1]
                    expected_j = [anchor2[0] + ox, anchor2[1] + oy]
                    for j in range(n):
                        if j == i:
                            continue
                        if cur[j][0] == expected_j[0] and cur[j][1] == expected_j[1]:
                            # Found a match: place new_pt + offset at some index
                            # Pick a point to replace (one that's not i or j)
                            new_x = new_pt[0] + ox
                            new_y = new_pt[1] + oy
                            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                                break
                            # Find an index to replace (prefer one not i or j)
                            replace_idx = None
                            for k in range(n):
                                if k != i and k != j:
                                    replace_idx = k
                                    break
                            if replace_idx is None:
                                break
                            new_points = [list(p) for p in cur]
                            new_points[replace_idx] = [new_x, new_y]
                            if not _valid(new_points):
                                break
                            score = _count_ordinary_lines(new_points)
                            delta = score - cur_score
                            if delta <= 0 or rng.random() < math.exp(-delta / T):
                                cur = new_points
                                cur_score = score
                                if score < best_score:
                                    best = [list(p) for p in cur]
                                    best_score = score
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            continue

        elif move_type < 0.80:
            # --- Pair-swap ---
            i, j = rng.sample(range(n), 2)
            new_points = [list(p) for p in cur]
            new_points[i], new_points[j] = list(cur[j]), list(cur[i])
            if not _valid(new_points):
                continue
            score = _count_ordinary_lines(new_points)
            delta = score - cur_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur = new_points
                cur_score = score
                if score < best_score:
                    best = [list(p) for p in cur]
                    best_score = score

        elif move_type < 0.92:
            # --- Pair relocate: move both to nearby positions ---
            i, j = rng.sample(range(n), 2)
            step = rng.choice([1, 2, 3, 5, 10])
            new_points = [list(p) for p in cur]
            new_points[i] = [cur[i][0] + rng.randint(-step, step),
                             cur[i][1] + rng.randint(-step, step)]
            new_points[j] = [cur[j][0] + rng.randint(-step, step),
                             cur[j][1] + rng.randint(-step, step)]
            if not _valid(new_points):
                continue
            score = _count_ordinary_lines(new_points)
            delta = score - cur_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur = new_points
                cur_score = score
                if score < best_score:
                    best = [list(p) for p in cur]
                    best_score = score

        else:
            # --- Large jump: reset one point to a random position ---
            idx = rng.randint(0, n - 1)
            new_x = rng.randint(-100, 100)
            new_y = rng.randint(-100, 100)
            new_points = [list(p) for p in cur]
            new_points[idx] = [new_x, new_y]
            if not _valid(new_points):
                continue
            score = _count_ordinary_lines(new_points)
            delta = score - cur_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur = new_points
                cur_score = score
                if score < best_score:
                    best = [list(p) for p in cur]
                    best_score = score

        T *= cooling

    return best, best_score


def _generate_seeds():
    """Generate diverse structured seeds."""
    seeds = []

    # 1. 3x4 grid
    seeds.append([[x, y] for x in range(4) for y in range(3)])

    # 2. 4x3 grid (transposed)
    seeds.append([[x, y] for x in range(3) for y in range(4)])

    # 3. Near-pencil: 11 on x-axis + 1 off
    seeds.append([[i, 0] for i in range(11)] + [[5, 1]])

    # 4. Two lines of 6, parallel
    seeds.append([[i, 0] for i in range(6)] + [[i, 3] for i in range(6)])

    # 5. Three parallel lines of 4
    seeds.append([[i, 0] for i in range(4)] + [[i, 2] for i in range(4)] + [[i, 4] for i in range(4)])

    # 6. Star: 4 lines of 3 through origin
    pts = []
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        for t in [-1, 1, 2]:
            p = [t * dx, t * dy]
            if p not in pts:
                pts.append(p)
    if len(pts) == 12:
        seeds.append(pts)

    # 7. Two intersecting lines sharing origin: 6+6
    pts = [[i, 0] for i in range(1, 7)] + [[0, i] for i in range(1, 7)]
    if len(pts) == 12:
        seeds.append(pts)

    # 8. Diagonal grid
    seeds.append([[i, j + i] for i in range(4) for j in range(3)])

    # 9. Parabolic
    seeds.append([[i, i * i] for i in range(12)])

    # 10. 5+4+3 parallel
    seeds.append([[i, 0] for i in range(5)] + [[i, 2] for i in range(4)] + [[i, 4] for i in range(3)])

    # 11. 6 on x-axis, 6 on diagonal y=x (sharing (0,0))
    pts = [[i, 0] for i in range(1, 7)] + [[i, i] for i in range(1, 7)]
    if len(pts) == 12:
        seeds.append(pts)

    # 12. Hexagonal-ish pattern
    pts = [[-2, -1], [-1, -2], [0, -2], [1, -2], [2, -1],
           [-2, 1], [-1, 2], [0, 2], [1, 2], [2, 1],
           [-1, 0], [1, 0]]
    if len(pts) == 12 and not _are_all_collinear(pts):
        seeds.append(pts)

    # 13. 4 collinear + 4 collinear + 4 collinear (three parallel lines, offset)
    seeds.append([[i, 0] for i in range(4)] + [[i+1, 3] for i in range(4)] + [[i+2, 6] for i in range(4)])

    # 14. Random seeds
    rng = random.Random(42)
    for trial in range(40):
        pts = []
        for _ in range(12):
            x = rng.randint(-5, 5)
            y = rng.randint(-5, 5)
            if [x, y] not in pts:
                pts.append([x, y])
        while len(pts) < 12:
            x = rng.randint(-10, 10)
            y = rng.randint(-10, 10)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 12 and not _are_all_collinear(pts):
            seeds.append(pts)

    # 15. Random with collinear bias
    rng = random.Random(123)
    for trial in range(30):
        dx = rng.randint(-3, 3)
        dy = rng.randint(-3, 3)
        if dx == 0 and dy == 0:
            dx, dy = 1, 0
        bx, by = rng.randint(-3, 3), rng.randint(-3, 3)
        k_on_line = rng.randint(5, 7)
        pts = []
        for t in range(k_on_line):
            p = [bx + t * dx, by + t * dy]
            if p not in pts:
                pts.append(p)
        while len(pts) < 12:
            x = rng.randint(-8, 8)
            y = rng.randint(-8, 8)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 12 and not _are_all_collinear(pts):
            seeds.append(pts)

    # 16. Two intersecting lines with 5+5 sharing a point, plus 2 more
    pts = [[i, 0] for i in range(1, 6)] + [[0, i] for i in range(1, 6)] + [[2, 2], [-1, 3]]
    if len(pts) == 12:
        seeds.append(pts)

    # 17. Configuration with 3 collinear triples on different lines
    # Line 1: (0,0),(1,0),(2,0)
    # Line 2: (0,1),(1,1),(2,1)
    # Line 3: (0,2),(1,2),(2,2)
    # Plus 3 more: (3,0),(3,1),(3,2) -> that's 3x4 grid
    # Instead: put them on non-parallel lines
    pts = [[0,0],[1,0],[2,0],  # line y=0
           [0,1],[2,1],[4,1],  # line y=1 (3 pts)
           [1,2],[3,2],[5,2]]  # line y=2 (3 pts)
    if len(pts) == 9:  # need 3 more
        pts += [[-1, 3], [0, 3], [1, 3]]
    if len(pts) == 12 and not _are_all_collinear(pts):
        seeds.append(pts)

    # 18. Known good: modified grid
    # Try a grid where one point is moved to create extra collinearity
    base = [[x, y] for x in range(4) for y in range(3)]
    # Move (3,2) to (5,2) to align with (0,2),(1,2),(2,2),(3,2)->(5,2)? No.
    # Move (3,2) to (0,3)? Then (0,0),(0,1),(0,2),(0,3) are collinear.
    mod = [[x, y] for x in range(4) for y in range(3) if (x, y) != (3, 2)] + [[0, 3]]
    if len(mod) == 12 and not _are_all_collinear(mod):
        seeds.append(mod)

    # Move (3,1) to (0,3)
    mod2 = [[x, y] for x in range(4) for y in range(3) if (x, y) != (3, 1)] + [[0, 3]]
    if len(mod2) == 12 and not _are_all_collinear(mod2):
        seeds.append(mod2)

    # Move (2,2) to (5,2) -> aligns (0,2),(1,2),(3,2),(5,2)? No, need (3,2) too.
    # Just move (3,2) to (5,2): now (0,2),(1,2),(2,2),(5,2) on y=2
    mod3 = [[x, y] for x in range(4) for y in range(3) if (x, y) != (3, 2)] + [[5, 2]]
    if len(mod3) == 12 and not _are_all_collinear(mod3):
        seeds.append(mod3)

    return seeds


def _polish(points, deadline):
    """Final high-precision polish: try all single-point moves to nearby lattice positions."""
    n = len(points)
    best = [list(p) for p in points]
    best_score = _count_ordinary_lines(best)
    
    if time.time() > deadline:
        return best, best_score

    # Try moving each point to nearby positions on a fine grid
    for idx in range(n):
        if time.time() > deadline:
            break
        ox, oy = best[idx]
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pts = [list(p) for p in best]
                new_pts[idx] = [nx, ny]
                if not _valid(new_pts):
                    continue
                s = _count_ordinary_lines(new_pts)
                if s < best_score:
                    best = new_pts
                    best_score = s
                    if best_score <= 6:
                        return best, best_score
    return best, best_score


def _try_construction():
    """Try various constructions and return the best configuration."""
    seeds = _generate_seeds()

    best_points = None
    best_score = float('inf')

    deadline = time.time() + 140  # Safety margin for 150s limit

    # Generate collinear triples from lattice for constraint-satisfaction moves
    triples = _generate_collinear_triples(lattice_size=16)

    # Score and sort seeds
    scored = []
    for s in seeds:
        if not _valid(s):
            continue
        sc = _count_ordinary_lines(s)
        scored.append((sc, s))
    scored.sort(key=lambda x: x[0])

    # Allocate search time: top seeds get more iterations
    n_seeds = len(scored)
    for rank, (init_score, seed) in enumerate(scored):
        if time.time() > deadline:
            break

        # Give more iterations to better seeds
        if rank < 3:
            max_iters = 6000
            n_restarts = 3
        elif rank < 8:
            max_iters = 3000
            n_restarts = 2
        elif rank < 15:
            max_iters = 1500
            n_restarts = 1
        else:
            max_iters = 800
            n_restarts = 1

        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            opt, score = _local_search(seed, max_iters=max_iters, deadline=deadline,
                                       seed=rank * 100 + restart, triples=triples)
            if score < best_score:
                best_score = score
                best_points = opt
            if best_score <= 6:
                return best_points, best_score

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)

    # Final polish pass
    if time.time() < deadline - 10:
        best_points, best_score = _polish(best_points, deadline)

    return best_points, best_score


def construct():
    """Construct 12 integer points minimizing ordinary lines."""
    points, score = _try_construction()

    # Ensure validity
    assert len(points) == 12
    assert len(set(tuple(p) for p in points)) == 12
    for p in points:
        assert abs(p[0]) <= 10**9
        assert abs(p[1]) <= 10**9
    assert not _are_all_collinear(points)

    return points

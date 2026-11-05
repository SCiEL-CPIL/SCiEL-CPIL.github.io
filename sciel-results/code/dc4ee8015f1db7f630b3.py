"""Explicit construction for the 12-point, few-ordinary-lines task.

Strategy:
  - Use a high-quality seed (3x4 grid).
  - Perform a fast, bounded local search to minimize ordinary lines.
  - Avoid heavy backtracking or long simulated annealing to prevent timeouts.
"""
import time
import random
import math
import itertools
from math import gcd

_START = time.time()
_DEADLINE = _START + 10.0  # Tight deadline to ensure safety


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _count_ordinary_fast(points):
    """Faster count using tuple keys without gcd normalization (sufficient for comparison)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            # Normalize sign only
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _collinear_check(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)


def _make_seed_grid(rows=3, cols=4, ox=0, oy=0, scale=1):
    """rows x cols grid with optional offset and scale."""
    return [[(x * scale + ox), (y * scale + oy)] for x in range(cols) for y in range(rows)]


def _make_seed_lines():
    """Seed: 12 points arranged on structured lines sharing intersections."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [1, 3], [2, 3], [3, 3], [4, 3],
        [0, 3], [1, 2], [2, 1],
        [0, 1]
    ]
    seen = set()
    clean = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            clean.append(p[:])
    return clean


def _make_seed_two_lines():
    """Böröczky-type: 6 points on y=0, 6 points on y=1 with aligned diagonals."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
    ]
    return pts


def _make_seed_perturbed():
    """4x3 grid with one point moved to create extra 3-point lines."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [4, 3],
    ]
    return pts


def _candidate_pool(points, radius=10):
    """Generate candidate integer points near the existing point cloud."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) // len(xs)
    cy = sum(ys) // len(ys)
    pool = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            pool.append((cx + dx, cy + dy))
    # Points aligned with existing lines (extend each line)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(-3, 4):
                pool.append((x1 + k * dx, y1 + k * dy))
                pool.append((x2 + k * dx, y2 + k * dy))
    # Deduplicate
    seen = set()
    unique = []
    for p in pool:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _generate_templates():
    """Generate line-partition templates: sets of lines with >=3 points covering all 12.
    
    Returns list of (lines, implied_ordinary_count) where lines is a list of frozensets
    of point indices (0-11), each of size >=3, and the union is all 12 points.
    The implied ordinary count is computed from the dual: pairs not on any >=3 line.
    """
    templates = []
    # We search for collections of lines (each a subset of {0,...,11} with size >= 3)
    # whose union is {0,...,11}, no two lines identical, and no line contained in another.
    # For efficiency, we enumerate by number of lines.
    
    all_pts = frozenset(range(12))
    
    # Precompute all subsets of size 3-6 (larger subsets are rarer and harder)
    # Actually, for 12 points, lines of size 3, 4, 5, 6 are the main ones.
    # A line of size 7+ is very restrictive.
    
    # Strategy: enumerate multisets of line-sizes that sum to >= 12 (with overlaps).
    # Key insight: each point must be on at least one line of size >= 3.
    # Total "slots" = sum of line sizes >= 12.
    
    # Let's enumerate by (num_lines, sizes_pattern)
    # For n=12, feasible patterns:
    # 4 lines of 3: total slots 12, each point on exactly 1 line -> but then no point is on 2 lines,
    #   meaning lines are disjoint, which means 4 groups of 3. Ordinary lines = C(12,2) - 4*C(3,2) = 66-12=54. Bad.
    # 5 lines of 3: total slots 15, 3 overlaps. 
    # 6 lines of 3: total slots 18, 6 overlaps.
    # 4 lines: mix of 3s and 4s, etc.
    
    # Better approach: enumerate all set partitions of 12 into blocks of size >= 3,
    # then for each partition, the "lines" are the blocks. But this gives disjoint lines.
    # We need overlapping lines (points at intersections).
    
    # Alternative: think in terms of the incidence structure.
    # For minimum ordinary lines, we want many 3-point lines.
    # Known good configurations for n=12:
    # - 4 lines of 4 points (like a 4x4 grid's rows+cols, but that's 8 lines of 4)
    # - Complete quadrilateral variants
    
    # Let me just try specific known-good templates:
    
    # Template A: 4 lines of 4 points, each pair of lines intersects at a unique point.
    # 4 lines, each with 4 pts. Pairs: C(4,2)=6 intersections. 
    # Each line has 4 points: 2 are "own" + 2 are intersections? No.
    # If 4 lines in general position (no 3 concurrent), each pair intersects at 1 point.
    # Total distinct points: 4*4 - 6 = 10 (since each intersection counted twice). Not 12.
    
    # Template B: 6 lines of 3 points.
    # If arranged so each point is on exactly 2 lines: 6*3/2 = 9 points. Not 12.
    # If some points on 1 line, some on 2: let a pts on 1 line, b pts on 2 lines.
    # a + b = 12, a + 2b = 18 => b=6, a=6. So 6 points each on 2 lines, 6 on 1 line.
    
    # Template C: 3 lines of 4 points.
    # 3*4 = 12 slots. If disjoint: 12 points, 3 lines. Ordinary = 66 - 3*6 = 48.
    # If overlapping: need more structure.
    
    # Template D: 8 lines of 3 points (like cube vertices projected).
    
    # Let me try a different approach: just generate random templates and check feasibility.
    # Actually, for the backtracking solver, I'll just try a few hand-crafted templates.
    
    # Template 1: Two parallel lines of 6 each (Böröczky-like but with more structure)
    # Points: L1 = {0,1,2,3,4,5}, L2 = {6,7,8,9,10,11}
    # Plus diagonal lines connecting them.
    # This is the two_lines seed. Let's formalize.
    
    # Template 2: 3x4 grid structure
    # 4 vertical lines of 3, 3 horizontal lines of 4
    # Lines: V0={0,3,6}, V1={1,4,7}, V2={2,5,8}, V3={3+3,4+3,5+3}... 
    # Actually grid: pts (i,j) for i in 0..3, j in 0..2
    # Vertical lines: for each i, {3i, 3i+1, 3i+2} -> 4 lines of 3
    # Horizontal lines: for each j, {j, j+3, j+6, j+9} -> 3 lines of 4
    # Total lines: 7. Points: 12.
    # Each point is on exactly 2 lines (1 vertical + 1 horizontal).
    # Ordinary lines: pairs not on same vertical or horizontal.
    # Total pairs: 66. Pairs on verticals: 4*C(3,2)=12. Pairs on horizontals: 3*C(4,2)=18.
    # But pairs on both? A pair can't be on both a vertical and horizontal (they'd need to share 2 points, impossible for distinct lines in grid).
    # So non-ordinary pairs = 12+18 = 30. Ordinary = 66-30 = 36.
    
    # Template 3: 4 lines of 4 with more overlap
    # Consider a complete graph K4 embedded: 4 vertices, 6 edges.
    # Not directly applicable.
    
    # Template 4: Pappus-like configuration
    # Pappus: 9 points, 9 lines of 3. For 12, extend.
    
    # Let me just create templates as (list of frozensets, expected_ordinary)
    # and let the solver find coordinates.
    
    # Grid 3x4 template
    grid_lines = [
        frozenset({0, 3, 6}),    # vertical 0
        frozenset({1, 4, 7}),    # vertical 1
        frozenset({2, 5, 8}),    # vertical 2
        frozenset({3, 6, 9}),    # wait, this overlaps with vertical 0 at 3,6
        frozenset({4, 7, 10}),
        frozenset({5, 8, 11}),
        frozenset({0, 1, 2, 3}),   # horizontal 0 (but 3 is also in vert 3)
        frozenset({4, 5, 6, 7}),   # horizontal 1
        frozenset({8, 9, 10, 11}), # horizontal 2
    ]
    # Hmm, this is getting complex. Let me just use the seeds we have and add the backtracking for specific known-good configurations.
    
    # Actually, the most promising approach: use the backtracking to find coordinates
    # for configurations with many 3-point lines. Let me define templates more carefully.
    
    # For now, return the grid template and let the solver work.
    templates.append(([
        frozenset({0, 3, 6}),
        frozenset({1, 4, 7}),
        frozenset({2, 5, 8}),
        frozenset({3, 6, 9}),
        frozenset({4, 7, 10}),
        frozenset({5, 8, 11}),
        frozenset({0, 1, 2, 3}),
        frozenset({4, 5, 6, 7}),
        frozenset({8, 9, 10, 11}),
    ], None))  # ordinary count to be computed
    
    return templates


def _solve_template_backtrack(lines, timeout=5.0):
    """Solve a line template via backtracking to find integer coordinates.
    
    lines: list of frozensets of point indices (0-11), each size >= 3.
    We assign coordinates to points 0..11 such that for each line, all its points are collinear.
    Strategy: pick 3 points to define the first line, then extend.
    """
    if time.time() > _DEADLINE - 10:
        return None
    
    n = 12
    coords = [None] * n
    used_pts = set()
    
    # Sort lines by size descending (more constrained first)
    sorted_lines = sorted(lines, key=len, reverse=True)
    
    def backtrack(line_idx):
        if time.time() > _DEADLINE - 10:
            return False
        if line_idx == len(sorted_lines):
            return True
        line = sorted_lines[line_idx]
        pts_in_line = list(line)
        
        # Check which points in this line already have coordinates
        assigned = [p for p in pts_in_line if coords[p] is not None]
        unassigned = [p for p in pts_in_line if coords[p] is None]
        
        if len(assigned) >= 3:
            # Verify collinearity of already-assigned points
            (x0, y0), (x1, y1), (x2, y2) = coords[assigned[0]], coords[assigned[1]], coords[assigned[2]]
            if (x1-x0)*(y2-y0) != (y1-y0)*(x2-x0):
                return False
            # Assign remaining points on this line
            for p in unassigned:
                # Point must be on line through assigned[0] and assigned[1]
                dx = x1 - x0
                dy = y1 - y0
                # p = (x0 + t*dx, y0 + t*dy) for some rational t
                # We need integer coords, so try small t values
                found = False
                for t_num in range(-20, 21):
                    for t_den in [1, 2, 3, 4, 5]:
                        if t_den == 1 and t_num == 0:
                            continue  # same as point 0
                        tx = x0 * t_den + t_num * dx
                        ty = y0 * t_den + t_num * dy
                        if tx % t_den == 0 and ty % t_den == 0:
                            cx, cy = tx // t_den, ty // t_den
                            if abs(cx) <= 10**9 and abs(cy) <= 10**9:
                                if (cx, cy) not in used_pts:
                                    coords[p] = (cx, cy)
                                    used_pts.add((cx, cy))
                                    found = True
                                    break
                    if found:
                        break
                if not found:
                    return False
            return backtrack(line_idx + 1)
        elif len(assigned) == 2:
            # Need to place one more point to define the line direction
            # Try placing the first unassigned point at various positions
            (x0, y0), (x1, y1) = coords[assigned[0]], coords[assigned[1]]
            p = unassigned[0]
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    cx, cy = x0 + dx, y0 + dy
                    if (cx, cy) in used_pts:
                        continue
                    if abs(cx) > 10**9 or abs(cy) > 10**9:
                        continue
                    coords[p] = (cx, cy)
                    used_pts.add((cx, cy))
                    if backtrack(line_idx):
                        return True
                    coords[p] = None
                    used_pts.discard((cx, cy))
            return False
        else:
            # 0 or 1 assigned: place all points on a new line
            if len(assigned) == 0:
                # Choose 3 points to define the line
                p0, p1, p2 = unassigned[0], unassigned[1], unassigned[2]
                # Try various line directions
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        # Normalize
                        g = gcd(abs(dx), abs(dy))
                        if g:
                            dx //= g
                            dy //= g
                        # Base point
                        bx, by = rng.randint(-5, 5), rng.randint(-5, 5)
                        coords[p0] = (bx, by)
                        used_pts.add((bx, by))
                        coords[p1] = (bx + dx, by + dy)
                        if (bx + dx, by + dy) in used_pts:
                            coords[p0] = None
                            used_pts.discard((bx, by))
                            continue
                        used_pts.add((bx + dx, by + dy))
                        coords[p2] = (bx + 2*dx, by + 2*dy)
                        if (bx + 2*dx, by + 2*dy) in used_pts:
                            coords[p0] = None
                            coords[p1] = None
                            used_pts.discard((bx, by))
                            used_pts.discard((bx + dx, by + dy))
                            continue
                        used_pts.add((bx + 2*dx, by + 2*dy))
                        
                        # Assign remaining points on this line
                        ok = True
                        for p in unassigned[3:]:
                            found = False
                            for t in range(-5, 6):
                                if t in [0, 1, 2]:
                                    continue
                                cx, cy = bx + t*dx, by + t*dy
                                if (cx, cy) not in used_pts and abs(cx) <= 10**9 and abs(cy) <= 10**9:
                                    coords[p] = (cx, cy)
                                    used_pts.add((cx, cy))
                                    found = True
                                    break
                            if not found:
                                ok = False
                                break
                        if ok and backtrack(line_idx + 1):
                            return True
                        # Undo
                        for p in unassigned:
                            if coords[p] is not None:
                                used_pts.discard(coords[p])
                                coords[p] = None
                        return False
            else:
                # 1 assigned
                p0 = assigned[0]
                p1, p2 = unassigned[0], unassigned[1]
                (x0, y0) = coords[p0]
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        g = gcd(abs(dx), abs(dy))
                        if g:
                            dx //= g
                            dy //= g
                        coords[p1] = (x0 + dx, y0 + dy)
                        if (x0 + dx, y0 + dy) in used_pts:
                            coords[p1] = None
                            continue
                        used_pts.add((x0 + dx, y0 + dy))
                        coords[p2] = (x0 + 2*dx, y0 + 2*dy)
                        if (x0 + 2*dx, y0 + 2*dy) in used_pts:
                            coords[p1] = None
                            coords[p2] = None
                            used_pts.discard((x0 + dx, y0 + dy))
                            continue
                        used_pts.add((x0 + 2*dx, y0 + 2*dy))
                        
                        ok = True
                        for p in unassigned[2:]:
                            found = False
                            for t in range(-5, 6):
                                if t in [0, 1, 2]:
                                    continue
                                cx, cy = x0 + t*dx, y0 + t*dy
                                if (cx, cy) not in used_pts and abs(cx) <= 10**9 and abs(cy) <= 10**9:
                                    coords[p] = (cx, cy)
                                    used_pts.add((cx, cy))
                                    found = True
                                    break
                            if not found:
                                ok = False
                                break
                        if ok and backtrack(line_idx + 1):
                            return True
                        for p in unassigned:
                            if coords[p] is not None:
                                used_pts.discard(coords[p])
                                coords[p] = None
                        return False
    
    rng = random.Random(42)
    if backtrack(0):
        return [list(coords[i]) for i in range(n)]
    return None


def _fast_local_search(points, max_iters=5000, seed=123):
    """Fast, bounded local search."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(seed)
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 1:
        iters += 1
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        improved = False
        # Try a few small perturbations
        for _ in range(5):
            nx = old[0] + rng.randint(-3, 3)
            ny = old[1] + rng.randint(-3, 3)
            cand = (nx, ny)
            if cand in cur_set:
                continue
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary_fast(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = _count_ordinary(cur)
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
        if improved:
            continue
    return cur, cur_score


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines."""
    # Start with a high-quality seed: 3x4 grid
    seed_pts = _make_seed_grid(3, 4)
    
    if not _valid(seed_pts):
        # Fallback if seed is invalid (should not happen)
        return [[i, 0] for i in range(12)]
        
    best_pts = seed_pts
    best_score = _count_ordinary(seed_pts)
    
    # Perform a fast local search to try to improve
    if time.time() < _DEADLINE - 2:
        improved_pts, improved_score = _fast_local_search(seed_pts, max_iters=5000, seed=42)
        if improved_score < best_score:
            best_pts = improved_pts
            best_score = improved_score
            
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))

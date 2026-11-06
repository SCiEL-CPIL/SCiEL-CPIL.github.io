import time
import random
from math import gcd

DEADLINE = 145

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', x1)
    elif dy == 0:
        return ('h', y1)
    else:
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0:
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * x1 - dx_n * y1
        return ('s', dx_n, dy_n, c)

def _build_lines(points):
    """Build line -> set of point indices. Single pass."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _check_no_four_collinear(lines):
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _get_triple_bitmasks(lines):
    """For each point i, bitmask of points j such that some k<j forms a triple with i,j... 
    Actually: for each pair (i,j), find if there's a k on the same line.
    We need: for a candidate subset, check no three collinear.
    Better: precompute for each i, the set of points that are collinear with i and some other point.
    Actually for branch and bound: when we have a partial subset S and want to add point i,
    we need to check that for all pairs (j,k) in S, i is not collinear with j and k.
    Equivalent: for each j in S, i must not be on the line through j and any k in S.
    
    Precompute: for each point i, and each other point j, the set of points k (k != i,j) 
    such that i,j,k are collinear. Then when adding i to S, for each j in S, 
    check if any k in S is in collinear_with[i][j].
    
    Simpler: precompute triple_set as set of frozensets of 3 indices.
    """
    triples = set()
    for pts in lines.values():
        if len(pts) >= 3:
            pts_list = sorted(pts)
            m = len(pts_list)
            for a in range(m):
                for b in range(a+1, m):
                    for c in range(b+1, m):
                        triples.add((pts_list[a], pts_list[b], pts_list[c]))
    return triples

def _max_cap_free_fast(points, triples, deadline):
    """Find largest subset with no three collinear using branch and bound."""
    n = len(points)
    triple_set = set(triples)
    best = [0]
    
    # Precompute: for each point i, list of (j, k) pairs where (min, mid, max) triple exists
    # When adding point i to a subset containing j and k, check if (i,j,k) is a triple
    # Faster: for each i, precompute a dict: for each j, bitmask of k's that form triple with i,j
    # But n=20, so bitmask fits in int
    
    # For each i, for each j != i, compute bitmask of k != i,j such that {i,j,k} is a triple
    collinear_mask = [[0]*n for _ in range(n)]
    for (a, b, c) in triple_set:
        collinear_mask[a][b] |= (1 << c)
        collinear_mask[a][c] |= (1 << b)
        collinear_mask[b][a] |= (1 << c)
        collinear_mask[b][c] |= (1 << a)
        collinear_mask[c][a] |= (1 << b)
        collinear_mask[c][b] |= (1 << a)
    
    def backtrack(subset_mask, start, current_size):
        if time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
        
        # Upper bound: can add at most (n - start) more
        if current_size + (n - start) <= best[0]:
            return
        
        for i in range(start, n):
            if time.time() > deadline:
                return
            if subset_mask & (1 << i):
                continue
            # Check if adding i is valid: for all j in subset, no k in subset with {i,j,k} triple
            valid = True
            # For each j in subset, check if collinear_mask[i][j] intersects subset
            temp = subset_mask
            while temp:
                j_bit = temp & (-temp)
                j = j_bit.bit_length() - 1
                if collinear_mask[i][j] & subset_mask:
                    valid = False
                    break
                temp -= j_bit
            
            if valid:
                backtrack(subset_mask | (1 << i), i+1, current_size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

def _compute_score(points, deadline):
    lines = _build_lines(points)
    if not _check_no_four_collinear(lines):
        return 20  # invalid, but shouldn't happen
    triples = _get_triple_bitmasks(lines)
    return _max_cap_free_fast(points, triples, deadline)

def _is_valid(points, deadline):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    lines = _build_lines(points)
    return _check_no_four_collinear(lines)

def _gen_parabola_perturbed(seed):
    """Generate points on a perturbed parabola."""
    rng = random.Random(seed)
    points = []
    xs = sorted(rng.sample(range(-50, 51), 20))
    for x in xs:
        y = x * x // 10 + rng.randint(-5, 5)
        y = max(-100, min(100, y))
        points.append([x, y])
    # Ensure distinct
    seen = set()
    unique = []
    for p in points:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    while len(unique) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    return unique[:20]

def _local_search(points, deadline):
    """Local search: try moving each point by small amounts."""
    best = [list(p) for p in points]
    best_score = _compute_score(best, deadline)
    
    for iteration in range(50):
        if time.time() > deadline - 15:
            break
        improved = False
        for i in range(20):
            if time.time() > deadline - 15:
                break
            orig = best[i][:]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    if (nx, ny) in [tuple(p) for p in best if p is not best[i]]:
                        continue
                    candidate = [list(p) for p in best]
                    candidate[i] = [nx, ny]
                    if _is_valid(candidate, deadline):
                        s = _compute_score(candidate, deadline)
                        if s < best_score:
                            best_score = s
                            best = candidate
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    random.seed(42)
    
    best_points = None
    best_score = 20
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Perturbed parabolas
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        pts = _gen_parabola_perturbed(seed)
        if _is_valid(pts, deadline):
            s = _compute_score(pts, deadline)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
    
    # Strategy 2: Random with local search
    for gen in range(30):
        if time.time() > deadline - 30:
            break
        pts = []
        while len(pts) < 20:
            x = random.randint(-80, 80)
            y = random.randint(-80, 80)
            if (x, y) not in [tuple(p) for p in pts]:
                pts.append([x, y])
        if _is_valid(pts, deadline):
            s = _compute_score(pts, deadline)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
    
    # Local search on best found
    if best_points is not None:
        if time.time() < deadline - 20:
            best_points, best_score = _local_search(best_points, deadline)
    
    # If no valid points found, use parabola baseline
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
        # Check validity, if not valid, try to fix
        if not _is_valid(best_points, deadline):
            # Use a different baseline
            best_points = []
            for x in range(-10, 10):
                y = x * x + (x % 3)
                best_points.append([x, y])
    
    return [list(p) for p in best_points]

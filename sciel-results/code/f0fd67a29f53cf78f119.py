import time
from math import isqrt
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sqdist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _collinear3(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def _conic_coeff(A, B, C, D):
    (x1, y1) = A
    (x2, y2) = B
    (x3, y3) = C
    (x4, y4) = D
    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3
    s4 = x4*x4 + y4*y4
    d = (
        x1 * y2 * (s3 - s4) - x1 * y3 * (s2 - s4) + x1 * y4 * (s2 - s3)
        - y1 * x2 * (s3 - s4) + y1 * x3 * (s2 - s4) - y1 * x4 * (s2 - s3)
        + s2 * (y3 * x4 - y4 * x3)
        - s3 * (y2 * x4 - y4 * x2)
        + s4 * (y2 * x3 - y3 * x2)
    )
    return d

def _valid_set(pts):
    if len(pts) != len(set(pts)):
        return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _is_square(_sqdist(pts[i], pts[j])):
                return False
    for a, b, c in combinations(pts, 3):
        if _collinear3(a, b, c) == 0:
            return False
    for a, b, c, d in combinations(pts, 4):
        if _conic_coeff(a, b, c, d) == 0:
            return False
    return True

def _format_points(pts):
    return {"D": 1, "points": [[[x, 1], [y, 1]] for (x, y) in pts]}

def _find_extension(current, deadline, search_radius=150):
    n = len(current)
    xs = [p[0] for p in current]
    ys = [p[1] for p in current]
    xmin, xmax = min(xs) - search_radius, max(xs) + search_radius
    ymin, ymax = min(ys) - search_radius, max(ys) + search_radius
    existing = set(current)
    
    # Precompute valid distance targets for each existing point
    # For each candidate, check all distances quickly
    
    for x in range(xmin, xmax + 1):
        if time.perf_counter() > deadline:
            return None
        for y in range(ymin, ymax + 1):
            if (x, y) in existing:
                continue
            cand = (x, y)
            
            # Check all squared distances are perfect squares
            ok = True
            for i in range(n):
                dx = x - current[i][0]
                dy = y - current[i][1]
                d2 = dx * dx + dy * dy
                if not _is_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            
            # Check no 3 collinear with any pair of existing
            bad = False
            for a in range(n):
                for b in range(a + 1, n):
                    if _collinear3(current[a], current[b], cand) == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            
            # Check no 4 concyclic with any triple of existing
            for a, b, c in combinations(range(n), 3):
                if _conic_coeff(current[a], current[b], current[c], cand) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            return cand
    
    return None

def _try_extend_iterative(base, deadline):
    """Iteratively extend a valid set by one point at a time."""
    current = base[:]
    
    while len(current) < 12:
        if time.perf_counter() > deadline:
            break
        
        # Try to find an extension with increasing search radius
        found = None
        for radius in [50, 100, 150, 200, 300]:
            if time.perf_counter() > deadline:
                break
            found = _find_extension(current, deadline, search_radius=radius)
            if found is not None:
                break
        
        if found is None:
            break
        
        new_set = current + [found]
        if _valid_set(new_set):
            current = new_set
        else:
            break
    
    return current

def _backtrack_search(points, target_size, deadline):
    """Backtracking search for a clique of given size in the valid-pair graph."""
    n_pts = len(points)
    
    # Build adjacency: which pairs have rational (square) distance
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if _is_square(_sqdist(points[i], points[j])):
                adj[i].add(j)
                adj[j].add(i)
    
    best = []
    
    def backtrack(current, current_set):
        if len(current) >= target_size:
            nonlocal best
            if len(current) > len(best):
                best = current[:]
            return
        
        if time.perf_counter() > deadline:
            return
        
        # Find candidates that connect to all in current
        candidates = []
        for i in range(n_pts):
            if i in current_set:
                continue
            ok = True
            for j in current:
                if i not in adj[j]:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check general position constraints
            cand = points[i]
            bad = False
            for a, b in combinations(current, 2):
                if _collinear3(points[a], points[b], cand) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            for a, b, c in combinations(current, 3):
                if _conic_coeff(points[a], points[b], points[c], cand) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            candidates.append(i)
        
        if not candidates:
            return
        
        # Sort candidates by degree (number of valid neighbors) for pruning
        candidates.sort(key=lambda i: len(adj[i]), reverse=True)
        
        for cand in candidates:
            if time.perf_counter() > deadline:
                return
            backtrack(current + [cand], current_set | {cand})
    
    # Try starting from each point
    for i in range(n_pts):
        if time.perf_counter() > deadline:
            break
        backtrack([i], {i})
        if len(best) >= target_size:
            break
    
    return best

def construct():
    start_time = time.perf_counter()
    deadline = start_time + 110  # 10 second safety margin
    
    # Known 5-point integral set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Validate base
    if not _valid_set(base):
        base = [(0, 0), (-56, 33), (-16, 63)]
    
    best = base[:]
    
    # Strategy 1: Iterative extension
    extended = _try_extend_iterative(base, deadline)
    if len(extended) > len(best):
        best = extended
    
    # Strategy 2: Backtracking search on a grid
    if time.perf_counter() < deadline - 20:
        # Create a grid of candidate points
        grid_points = []
        for x in range(-80, 81):
            for y in range(-80, 81):
                grid_points.append((x, y))
        
        # Try to find a set of 6 or more via backtracking
        found = _backtrack_search(grid_points, 6, deadline)
        if len(found) > len(best):
            best = [grid_points[i] for i in found]
    
    # Strategy 3: Try different base configurations
    if time.perf_counter() < deadline - 10:
        alt_bases = [
            [(0, 0), (9, 12), (12, 9), (21, 28), (28, 21)],
            [(0, 0), (-9, -12), (-12, -9), (-21, -28), (-28, -21)],
            [(0, 0), (5, 12), (12, 5), (15, 8), (8, 15)],
        ]
        for alt in alt_bases:
            if time.perf_counter() > deadline:
                break
            if _valid_set(alt):
                ext = _try_extend_iterative(alt, deadline)
                if len(ext) > len(best):
                    best = ext
    
    return _format_points(best)

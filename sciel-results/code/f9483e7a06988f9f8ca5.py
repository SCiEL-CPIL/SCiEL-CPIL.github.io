import time
import random
from math import gcd

def _collinear3(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        dy = 1
    elif dy == 0:
        dx = 1
    else:
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
    c = dx * y1 - dy * x1
    return (dx, dy, c)

def _has_four_collinear(points):
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for pts in line_map.values():
        if len(pts) >= 4:
            return True
    return False

def _count_collinear_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_3free_subset(points, deadline=None):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    # Precompute: for each pair (i,j), which other points are collinear with them
    collinear_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if i == k or j == k:
                    continue
                if _collinear3(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask
            collinear_mask[j][i] = mask

    best = [0]

    def can_add(i, current_mask):
        """Check if point i can be added to the current subset."""
        temp = current_mask
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            temp -= lsb
            if (collinear_mask[i][j] & current_mask) != 0:
                return False
    return False

    def dfs(current_mask, candidates_mask, current_size, depth):
        if deadline and time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
        if candidates_mask == 0:
            return
        # Upper bound
        remaining = bin(candidates_mask).count('1')
        if current_size + remaining <= best[0]:
            return
        # Pick next candidate
        lsb = candidates_mask & -candidates_mask
        i = lsb.bit_length() - 1
        new_cands = candidates_mask & ~lsb

        # Branch: skip i
        dfs(current_mask, new_cands, current_size, depth + 1)

        # Branch: include i (if possible)
        if can_add(i, current_mask):
            new_current = current_mask | lsb
            dfs(new_current, new_cands, current_size + 1, depth + 1)

    dfs(0, (1 << n) - 1, 0, 0)
    return best[0]

def _construct_line_packed(deadline=None):
    """Systematically construct 20 points with many 3-point collinear triples."""
    points = []
    seen = set()

    # Start with a base line of 3 points
    base_x, base_y = 0, 0
    p0 = [base_x, base_y]
    p1 = [base_x + 3, base_y]
    p2 = [base_x + 6, base_y]
    for p in [p0, p1, p2]:
        if abs(p[0]) <= 100 and abs(p[1]) <= 100:
            points.append(p)
            seen.add(tuple(p))

    # Add points to maximize new collinear triples
    while len(points) < 20:
        if deadline and time.time() > deadline:
            break
        best_candidate = None
        best_gain = -1

        # Try positions that extend existing lines or create new 3-point lines
        candidates = []

        # Extend existing lines
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for t in [-3, -2, -1, 1, 2, 3, 4]:
                    if t == 0:
                        continue
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    cand = [nx, ny]
                    if tuple(cand) not in seen:
                        candidates.append(cand)

        # Try random positions
        for _ in range(100):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            cand = [x, y]
            if tuple(cand) not in seen:
                candidates.append(cand)

        # Evaluate candidates
        for cand in candidates:
            if tuple(cand) in seen:
                continue
            test_points = points + [cand]
            if _has_four_collinear(test_points):
                continue
            new_triples = 0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    if _collinear3(points[i], points[j], cand):
                        new_triples += 1
            if new_triples > best_gain:
                best_gain = new_triples
                best_candidate = cand

        if best_candidate:
            points.append(best_candidate)
            seen.add(tuple(best_candidate))
        else:
            # Fallback: add random valid point
            for _ in range(200):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                if tuple(cand) in seen:
                    continue
                test_points = points + [cand]
                if _has_four_collinear(test_points):
                    continue
                points.append(cand)
                seen.add(tuple(cand))
                break
            else:
                # Force add
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                points.append(cand)
                seen.add(tuple(cand))

    return points[:20]

def _local_search_line_aware(points, deadline):
    """Local search to improve collinear triple count."""
    current = [p[:] for p in points]
    current_triples = _count_collinear_triples(current)
    best = [p[:] for p in current]
    best_triples = current_triples

    while time.time() < deadline:
        idx = random.randint(0, 19)
        orig = current[idx][:]
        moved = False

        # Try moving to positions on lines through pairs
        for _ in range(40):
            i = random.randint(0, 19)
            j = random.randint(0, 19)
            if i == j or i == idx or j == idx:
                continue
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            for t in random.sample([-4, -3, -2, -1, 1, 2, 3, 4], 2):
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) > 100 or abs(ny) > 100:
                    continue
                cand = [nx, ny]
                if any(tuple(cand) == tuple(p) for k, p in enumerate(current) if k != idx):
                    continue
                test = current[:]
                test[idx] = cand
                if _has_four_collinear(test):
                    continue
                new_triples = _count_collinear_triples(test)
                if new_triples > current_triples:
                    current[idx] = cand
                    current_triples = new_triples
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best = [p[:] for p in current]
                    moved = True
                    break
            if moved:
                break

        if not moved:
            # Try random move
            for _ in range(10):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                if any(tuple(cand) == tuple(p) for k, p in enumerate(current) if k != idx):
                    continue
                test = current[:]
                test[idx] = cand
                if _has_four_collinear(test):
                    continue
                new_triples = _count_collinear_triples(test)
                if new_triples > current_triples:
                    current[idx] = cand
                    current_triples = new_triples
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best = [p[:] for p in current]
                    moved = True
                    break

        if not moved:
            current[idx] = orig

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    best_points = None
    best_score = 20

    # Try multiple line-packing constructions with local search
    for attempt in range(8):
        if time.time() > deadline - 15:
            break
        pts = _construct_line_packed(deadline - 10)
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue

        # Local search
        improved = _local_search_line_aware(pts, deadline - 5)

        # Evaluate with exact branch and bound
        score = _max_3free_subset(improved, deadline - 2)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in improved]

    if best_points is None:
        # Fallback: simple random construction
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if tuple(p) in seen:
                continue
            points.append(p)
            if _has_four_collinear(points):
                points.pop()
                continue
            seen.add(tuple(p))
        best_points = points

    return best_points

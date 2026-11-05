import time
import random
from itertools import combinations

def _collinear3(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _has_four_collinear(points):
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return a
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
            c = dx * y1 - dy * x1
            key = (dx, dy, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts in line_map.items():
        if len(pts) >= 4:
            return True
    return False

def _count_collinear_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_3free_subset(points, deadline=None):
    n = len(points)
    collinear_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            mask = 0
            for k in range(n):
                if i == k or j == k: continue
                if _collinear3(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask

    best = [0]
    
    def can_add(i, current_mask):
        temp = current_mask
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            temp -= lsb
            if (collinear_mask[i][j] & current_mask) != 0:
                return False
        return True

    def dfs(current_mask, candidates_mask, current_size):
        if deadline and time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
        if candidates_mask == 0:
            return
        if current_size + bin(candidates_mask).count('1') <= best[0]:
            return
        lsb = candidates_mask & -candidates_mask
        i = lsb.bit_length() - 1
        dfs(current_mask, candidates_mask & ~lsb, current_size)
        if can_add(i, current_mask):
            new_current = current_mask | lsb
            dfs(new_current, candidates_mask & ~lsb, current_size + 1)

    dfs(0, (1 << n) - 1, 0)
    return best[0]

def _try_line_packing(deadline=None):
    """Systematically build points to create 3-point lines."""
    points = []
    seen = set()
    
    # Start with 3 points forming a line
    base_x, base_y = random.randint(-50, 50), random.randint(-50, 50)
    dx, dy = random.choice([(2,0),(0,2),(1,1),(1,-1),(2,1),(1,2)])
    p0 = [base_x, base_y]
    p1 = [base_x + dx, base_y + dy]
    p2 = [base_x + 2*dx, base_y + 2*dy]
    for p in [p0, p1, p2]:
        if abs(p[0]) <= 100 and abs(p[1]) <= 100 and tuple(p) not in seen:
            points.append(p)
            seen.add(tuple(p))
    
    # Add more points, preferring positions that create new 3-point lines
    while len(points) < 20:
        if deadline and time.time() > deadline:
            break
        best_candidate = None
        best_gain = -1
        
        # Try random positions first
        for _ in range(50):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            cand = [x, y]
            if tuple(cand) in seen:
                continue
            if abs(x) > 100 or abs(y) > 100:
                continue
            
            # Check no 4 collinear
            test_points = points + [cand]
            if _has_four_collinear(test_points):
                continue
            
            # Count how many new triples this creates
            new_triples = 0
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if _collinear3(points[i], points[j], cand):
                        new_triples += 1
            
            if new_triples > best_gain:
                best_gain = new_triples
                best_candidate = cand
        
        # Also try positions on lines through pairs of existing points
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Try extending the line
                for t in [-2, -1, 1, 2, 3]:
                    if t == 0: continue
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    cand = [nx, ny]
                    if tuple(cand) in seen:
                        continue
                    test_points = points + [cand]
                    if _has_four_collinear(test_points):
                        continue
                    new_triples = 0
                    for a in range(len(points)):
                        for b in range(a+1, len(points)):
                            if _collinear3(points[a], points[b], cand):
                                new_triples += 1
                    if new_triples > best_gain:
                        best_gain = new_triples
                        best_candidate = cand
        
        if best_candidate:
            points.append(best_candidate)
            seen.add(tuple(best_candidate))
        else:
            # Fallback: random
            for _ in range(100):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                if tuple(cand) in seen or abs(x) > 100 or abs(y) > 100:
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
    """Local search that specifically tries to create new collinear triples."""
    current = [p[:] for p in points]
    current_triples = _count_collinear_triples(current)
    best = [p[:] for p in current]
    best_triples = current_triples
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        orig = current[idx][:]
        
        # Try to move it to a position that creates new triples
        moved = False
        for _ in range(30):
            # Try positions on lines through pairs
            i = random.randint(0, 19)
            j = random.randint(0, 19)
            if i == j or i == idx or j == idx:
                continue
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            for t in random.sample([-3,-2,-1,1,2,3], 2):
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
            for _ in range(5):
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
    deadline = start_time + 115
    
    best_points = None
    best_score = 20
    
    # Try multiple line-packing constructions
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        pts = _try_line_packing(deadline - 5)
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        
        # Do local search to improve
        improved = _local_search_line_aware(pts, deadline - 5)
        
        # Evaluate
        score = _max_3free_subset(improved, deadline - 2)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in improved]
    
    if best_points is None:
        # Fallback: random construction
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

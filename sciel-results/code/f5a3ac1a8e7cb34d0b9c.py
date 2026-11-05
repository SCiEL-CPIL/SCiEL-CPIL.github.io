import time
import random
from itertools import combinations
from math import gcd

def _collinear3(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _line_key(p1, p2):
    """Return normalized line key for two points."""
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
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts in line_map.items():
        if len(pts) >= 4:
            return True
    return False

def _count_collinear_triples(points):
    """Count collinear triples using line grouping for efficiency."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    count = 0
    for key, pts in line_map.items():
        k = len(pts)
        if k >= 3:
            count += k * (k-1) * (k-2) // 6
    return count

def _max_3free_subset(points, deadline=None):
    """Find largest subset with no three collinear using branch and bound."""
    n = len(points)
    if n <= 1:
        return n
    
    # Precompute for each pair (i,j) which points are collinear with both
    # collinear_with[i][j] = bitmask of k where i,j,k are collinear
    collinear_with = {}
    for i in range(n):
        for j in range(i+1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _collinear3(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_with[(i,j)] = mask
            collinear_with[(j,i)] = mask
    
    best = [0]
    start_time = time.time()
    
    def can_add(i, current_mask):
        """Check if point i can be added to current set without creating 3 collinear."""
        temp = current_mask
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            temp ^= lsb
            if (collinear_with[(i,j)] & current_mask) != 0:
                return False
        return True
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(current_mask, candidates_mask, current_size, depth):
        if deadline and time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
        if candidates_mask == 0:
            return
        if current_size + popcount(candidates_mask) <= best[0]:
            return
        
        # Choose next candidate: try to add points that create most constraints
        # to prune earlier
        lsb = candidates_mask & -candidates_mask
        i = lsb.bit_length() - 1
        
        # Branch 1: skip i
        dfs(current_mask, candidates_mask & ~lsb, current_size, depth + 1)
        
        # Branch 2: add i if possible
        if can_add(i, current_mask):
            new_current = current_mask | lsb
            # Remove points that would conflict with i given current set
            conflict_mask = 0
            temp = current_mask
            while temp:
                lsb2 = temp & -temp
                j = lsb2.bit_length() - 1
                temp ^= lsb2
                conflict_mask |= collinear_with[(i,j)]
            new_candidates = candidates_mask & ~lsb & ~conflict_mask
            dfs(new_current, new_candidates, current_size + 1, depth + 1)
    
    dfs(0, (1 << n) - 1, 0, 0)
    return best[0]

def _try_line_packing(deadline=None):
    """Systematically build points to create 3-point lines."""
    points = []
    seen = set()
    
    # Start with a small structure that has some collinear triples
    # Use a grid-like pattern with deliberate 3-point lines
    base_configs = [
        # Config 1: two intersecting lines
        [[0,0], [1,0], [2,0], [0,1], [0,2]],
        # Config 2: triangle with midpoints
        [[0,0], [2,0], [1,1], [0,2], [2,2]],
        # Config 3: star pattern
        [[0,0], [1,0], [-1,0], [0,1], [0,-1]],
    ]
    
    base = random.choice(base_configs)
    offset_x = random.randint(-30, 30)
    offset_y = random.randint(-30, 30)
    
    for p in base:
        np_ = [p[0] + offset_x, p[1] + offset_y]
        if abs(np_[0]) <= 100 and abs(np_[1]) <= 100 and tuple(np_) not in seen:
            points.append(np_)
            seen.add(tuple(np_))
    
    # Add more points, preferring positions that create new 3-point lines
    while len(points) < 20:
        if deadline and time.time() > deadline:
            break
        best_candidate = None
        best_gain = -1
        current_triples = _count_collinear_triples(points)
        
        # Try positions on lines through pairs of existing points (most promising)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Try extending the line in both directions
                for t in [-3, -2, -1, 1, 2, 3]:
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
                    new_triples = _count_collinear_triples(test_points)
                    gain = new_triples - current_triples
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = cand
        
        # Also try random positions
        for _ in range(30):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            cand = [x, y]
            if tuple(cand) in seen:
                continue
            test_points = points + [cand]
            if _has_four_collinear(test_points):
                continue
            new_triples = _count_collinear_triples(test_points)
            gain = new_triples - current_triples
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
        
        if best_candidate:
            points.append(best_candidate)
            seen.add(tuple(best_candidate))
        else:
            # Fallback: find any valid position
            found = False
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
                found = True
                break
            if not found:
                # Force add a point far from others
                for x in range(-100, 101, 7):
                    for y in range(-100, 101, 7):
                        cand = [x, y]
                        if tuple(cand) in seen:
                            continue
                        test_points = points + [cand]
                        if _has_four_collinear(test_points):
                            continue
                        points.append(cand)
                        seen.add(tuple(cand))
                        found = True
                        break
                    if found:
                        break
                if not found:
                    break
    
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
        best_candidate = None
        best_gain = -1
        
        # Try positions on lines through pairs of other points
        for i in range(20):
            if i == idx:
                continue
            for j in range(i+1, 20):
                if j == idx:
                    continue
                x1, y1 = current[i]
                x2, y2 = current[j]
                dx = x2 - x1
                dy = y2 - y1
                for t in [-3, -2, -1, 1, 2, 3]:
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
                    gain = new_triples - current_triples
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = cand
        
        # Also try some random positions
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
            gain = new_triples - current_triples
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
        
        if best_candidate and best_gain > 0:
            current[idx] = best_candidate
            current_triples += best_gain
            if current_triples > best_triples:
                best_triples = current_triples
                best = [p[:] for p in current]
            moved = True
        
        if not moved:
            # Try accepting equal or slightly worse moves to escape local optima
            for _ in range(5):
                i = random.randint(0, 19)
                j = random.randint(0, 19)
                if i == j or i == idx or j == idx:
                    continue
                x1, y1 = current[i]
                x2, y2 = current[j]
                dx = x2 - x1
                dy = y2 - y1
                for t in random.sample([-3,-2,-1,1,2,3], 3):
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
                    if new_triples >= current_triples:
                        current[idx] = cand
                        current_triples = new_triples
                        if current_triples > best_triples:
                            best_triples = current_triples
                            best = [p[:] for p in current]
                        moved = True
                        break
                if moved:
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_score = 20
    
    # Try multiple line-packing constructions with different seeds
    random.seed()
    
    for attempt in range(8):
        if time.time() > deadline - 15:
            break
        
        # Reset random seed for reproducibility within attempt
        random.seed(attempt * 12345 + 67890)
        
        pts = _try_line_packing(deadline - 10)
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        
        # Do local search to improve
        ls_deadline = deadline - 8
        improved = _local_search_line_aware(pts, ls_deadline)
        
        # Evaluate
        score = _max_3free_subset(improved, deadline - 3)
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

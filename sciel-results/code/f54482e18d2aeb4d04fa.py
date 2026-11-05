import time
import random
from math import gcd

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
    """Fast exact B&B for maximum 3-free subset using bitmasks."""
    n = len(points)
    # collinear_mask[i][j]: bitmask of points k (k!=i,j) collinear with i and j
    collinear_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if _collinear3(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask
            collinear_mask[j][i] = mask

    best = [0]
    deadline_val = deadline if deadline else float('inf')

    def popcount(x):
        return bin(x).count('1')

    def dfs(current_mask, candidates_mask, current_size, depth):
        if time.time() > deadline_val:
            return
        if current_size > best[0]:
            best[0] = current_size
        if candidates_mask == 0:
            return
        # Upper bound
        if current_size + popcount(candidates_mask) <= best[0]:
            return

        # Pick vertex with max "conflict degree" in candidates for better pruning
        # For each candidate i, count how many candidates j would be blocked if i is added
        # i blocks j if collinear_mask[i][j] & current_mask != 0 (j forms triple with i and some in current)
        # Also, after adding i, j is blocked if collinear_mask[j][i] & (current_mask | (1<<i)) != 0
        # Simpler: pick first candidate for speed, or use a heuristic
        
        # Heuristic: pick candidate that maximizes immediate pruning
        # For candidate i, the number of candidates that become invalid if i is added:
        # A candidate k is invalid if (collinear_mask[k][i] & (current_mask | (1<<i))) != 0
        #   OR there exists u in current_mask such that (collinear_mask[k][u] & (current_mask | (1<<i))) != 0
        # This is expensive. Use simpler heuristic: pick the candidate with most collinear partners in candidates.
        
        # For speed, just pick LSB
        lsb = candidates_mask & -candidates_mask
        i = lsb.bit_length() - 1

        # Branch 1: Exclude i
        dfs(current_mask, candidates_mask & ~lsb, current_size, depth+1)

        # Branch 2: Include i (if compatible)
        # Check if i is compatible with current_mask:
        # For all j in current_mask, (collinear_mask[i][j] & current_mask) == 0
        can_add = True
        temp = current_mask
        while temp:
            lsb_c = temp & -temp
            j = lsb_c.bit_length() - 1
            temp -= lsb_c
            if (collinear_mask[i][j] & current_mask) != 0:
                can_add = False
                break

        if can_add:
            new_current = current_mask | lsb
            # Filter candidates: remove k if incompatible with new_current
            # k is incompatible if for some u in new_current, (collinear_mask[k][u] & new_current) != 0
            new_candidates = candidates_mask & ~lsb
            valid_cands = 0
            temp_c = new_candidates
            while temp_c:
                lsb_k = temp_c & -temp_c
                k = lsb_k.bit_length() - 1
                temp_c -= lsb_k
                compatible = True
                temp_u = new_current
                while temp_u:
                    lsb_u = temp_u & -temp_u
                    u = lsb_u.bit_length() - 1
                    temp_u -= lsb_u
                    if (collinear_mask[k][u] & new_current) != 0:
                        compatible = False
                        break
                if compatible:
                    valid_cands |= lsb_k
            dfs(new_current, valid_cands, current_size + 1, depth+1)

    dfs(0, (1 << n) - 1, 0, 0)
    return best[0]

def _construct_line_packed():
    """Construct a configuration with many 3-point lines to minimize max 3-free subset."""
    # Strategy: Use a structured approach based on finite field geometry ideas
    # Place points to create many collinear triples while avoiding 4-collinear
    
    points = []
    seen = set()
    
    # Phase 1: Build a core of points on shared lines
    # Use lines with small slopes to create overlapping triples
    
    # Define target lines (directions) we want to use
    # Each line will have exactly 3 points
    
    # Start with a small structured set
    base_points = [
        [0, 0], [1, 0], [2, 0],   # horizontal
        [0, 1], [1, 1], [2, 1],   # horizontal  
        [0, 2], [1, 2], [2, 2],   # horizontal
    ]
    
    # That's already 9 points with 3 horizontal lines of 3.
    # But we need to add more and create diagonal lines too.
    
    # Let me try a different approach: use a known good configuration
    # and perturb it.
    
    # Known approach: take points from a grid and select to maximize triples
    candidates = []
    for x in range(-15, 16):
        for y in range(-15, 16):
            candidates.append((x, y))
    
    random.shuffle(candidates)
    
    # Greedy: add point that creates most new triples
    for cand in candidates:
        if len(points) >= 20:
            break
        x, y = cand
        if (x, y) in seen:
            continue
        if abs(x) > 100 or abs(y) > 100:
            continue
        
        test_points = points + [[x, y]]
        if _has_four_collinear(test_points):
            continue
        
        new_triples = 0
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if _collinear3(points[i], points[j], [x, y]):
                    new_triples += 1
        
        points.append([x, y])
        seen.add((x, y))
    
    # Fill remaining if needed
    if len(points) < 20:
        attempts = 0
        while len(points) < 20 and attempts < 5000:
            attempts += 1
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) in seen:
                continue
            test_points = points + [[x, y]]
            if _has_four_collinear(test_points):
                continue
            points.append([x, y])
            seen.add((x, y))
    
    return points[:20]

def _local_search(points, deadline):
    """Local search to maximize collinear triples."""
    current = [p[:] for p in points]
    current_triples = _count_collinear_triples(current)
    best = [p[:] for p in current]
    best_triples = current_triples
    no_improve = 0
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        orig = current[idx][:]
        moved = False
        
        # Try to place on a line through two existing points
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
                for t in range(-8, 9):
                    if t == 0 or t == 1:
                        continue
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    cand = (nx, ny)
                    if cand in [tuple(p) for p in current]:
                        continue
                    test = [p[:] for p in current]
                    test[idx] = [nx, ny]
                    if _has_four_collinear(test):
                        continue
                    new_triples = _count_collinear_triples(test)
                    if new_triples > current_triples:
                        current[idx] = [nx, ny]
                        current_triples = new_triples
                        if current_triples > best_triples:
                            best_triples = current_triples
                            best = [p[:] for p in current]
                            no_improve = 0
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        
        if not moved:
            no_improve += 1
            if no_improve > 50:
                # Random restart perturbation
                for _ in range(3):
                    ridx = random.randint(0, 19)
                    x = random.randint(-80, 80)
                    y = random.randint(-80, 80)
                    if (x, y) in [tuple(p) for p in current]:
                        continue
                    test = [p[:] for p in current]
                    test[ridx] = [x, y]
                    if _has_four_collinear(test):
                        continue
                    current = test
                    current_triples = _count_collinear_triples(current)
                    no_improve = 0
                    break
                else:
                    break
            current[idx] = orig
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_score = 20
    
    # Try multiple constructions
    for attempt in range(8):
        if time.time() > deadline - 20:
            break
        
        pts = _construct_line_packed()
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        
        improved = _local_search(pts, deadline - 15)
        
        score = _max_3free_subset(improved, deadline - 5)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in improved]
        
        # If we found a good score, try to improve further
        if best_score <= 12:
            break
    
    if best_points is None:
        # Fallback
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

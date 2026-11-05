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

        # Heuristic: pick candidate that maximizes immediate pruning
        # Impact of adding i: number of k in candidates such that (collinear_mask[k][i] & current_mask) != 0
        best_i = -1
        best_lsb = 0
        best_impact = -1
        
        temp_c = candidates_mask
        while temp_c:
            lsb_i = temp_c & -temp_c
            i = lsb_i.bit_length() - 1
            temp_c -= lsb_i
            
            impact = 0
            temp_k = candidates_mask & ~lsb_i
            while temp_k:
                lsb_k = temp_k & -temp_k
                k = lsb_k.bit_length() - 1
                temp_k -= lsb_k
                if (collinear_mask[k][i] & current_mask) != 0:
                    impact += 1
            
            if impact > best_impact:
                best_impact = impact
                best_i = i
                best_lsb = lsb_i
        
        if best_i == -1:
            return

        i = best_i
        lsb = best_lsb

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
            # Since k was compatible with current_mask, we only need to check against i
            # k is incompatible with i if (collinear_mask[k][i] & current_mask) != 0
            new_candidates = candidates_mask & ~lsb
            valid_cands = 0
            temp_c = new_candidates
            while temp_c:
                lsb_k = temp_c & -temp_c
                k = lsb_k.bit_length() - 1
                temp_c -= lsb_k
                if (collinear_mask[k][i] & current_mask) == 0:
                    valid_cands |= lsb_k
            
            dfs(new_current, valid_cands, current_size + 1, depth+1)

    dfs(0, (1 << n) - 1, 0, 0)
    return best[0]

def _construct_line_packed():
    """Construct a configuration with many 3-point lines to minimize max 3-free subset."""
    points = []
    seen = set()
    
    # Use a structured grid as a starting point, then perturb
    # A 5x4 grid has 20 points.
    # We need to ensure no 4 collinear.
    # A standard grid has many 4-collinear lines.
    # We will generate a grid and then randomly move points to break 4-collinearities
    # while trying to maintain or increase triples.
    
    # Start with a random set of 20 points
    candidates = []
    for x in range(-15, 16):
        for y in range(-15, 16):
            candidates.append((x, y))
    
    random.shuffle(candidates)
    
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
        
        points.append([x, y])
        seen.add((x, y))
    
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
        # Steepest Descent: Find the best move
        best_move = None
        best_gain = 0
        
        for idx in range(20):
            orig = current[idx][:]
            # Try moving point idx to a new position
            # Generate candidate positions:
            # 1. On lines defined by pairs of other points
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
                    # Check a few offsets
                    for t in range(-3, 4):
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
                        gain = new_triples - current_triples
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (idx, [nx, ny])
                            
            # Also try random positions
            for _ in range(10):
                nx = random.randint(-80, 80)
                ny = random.randint(-80, 80)
                cand = (nx, ny)
                if cand in [tuple(p) for p in current]:
                    continue
                test = [p[:] for p in current]
                test[idx] = [nx, ny]
                if _has_four_collinear(test):
                    continue
                new_triples = _count_collinear_triples(test)
                gain = new_triples - current_triples
                if gain > best_gain:
                    best_gain = gain
                    best_move = (idx, [nx, ny])
                    
        if best_move:
            idx, new_pos = best_move
            current[idx] = new_pos
            current_triples += best_gain
            if current_triples > best_triples:
                best_triples = current_triples
                best = [p[:] for p in current]
                no_improve = 0
        else:
            no_improve += 1
            if no_improve > 20:
                # Random restart
                for _ in range(5):
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
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_score = 20
    
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        
        pts = _construct_line_packed()
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        
        improved = _local_search(pts, deadline - 25)
        
        score = _max_3free_subset(improved, deadline - 5)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in improved]
        
        if best_score <= 10:
            break
    
    if best_points is None:
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

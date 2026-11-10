import time
import random
from math import gcd

_START_TIME = None
_DEADLINE = 145.0  # Safe margin before 150s

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return 0.0
    return time.time() - _START_TIME

def _check_no_four_collinear(points):
    """Returns True if no 4 points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _get_triples(points):
    """
    Returns list of triples (i, j, k) of indices that are collinear.
    Also returns partner matrix: partner[i][j] = k if (i,j,k) collinear, else -1.
    """
    n = len(points)
    triples = []
    partner = [[-1] * n for _ in range(n)]
    seen_lines = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            
            # Normalize line key
            if dx == 0:
                a, b, c = 1, 0, -x1
            elif dy == 0:
                a, b, c = 0, 1, -y1
            else:
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                a, b = dy_n, -dx_n
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
            
            line_key = (a, b, c)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            
            # Find all points on this line
            line_pts = []
            for k in range(n):
                x3, y3 = points[k]
                if a * x3 + b * y3 + c == 0:
                    line_pts.append(k)
            
            m = len(line_pts)
            if m >= 3:
                # Generate all triples
                for p in range(m):
                    for q in range(p + 1, m):
                        for r in range(q + 1, m):
                            i, j, k = line_pts[p], line_pts[q], line_pts[r]
                            triples.append((i, j, k))
                            partner[i][j] = k
                            partner[i][k] = j
                            partner[j][i] = k
                            partner[j][k] = i
                            partner[k][i] = j
                            partner[k][j] = i
    return triples, partner

def _max_independent_set_size(partner, n):
    """
    Exact branch and bound for the largest subset with no three collinear.
    Input: partner matrix, n (total points)
    Output: Size of the largest subset with no complete triple.
    """
    # Bitset operations
    # S: current set
    # candidates: remaining points
    best = [0]
    
    # Precompute popcount for 0 to 2^20? No, too big. Use int.bit_count() (Python 3.10+)
    # If older Python, use bin(x).count('1')
    
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
        
    def backtrack(S, candidates):
        if _elapsed() > _DEADLINE - 2:
            return
        
        # Upper bound
        if popcount(S) + popcount(candidates) <= best[0]:
            return
            
        if candidates == 0:
            if popcount(S) > best[0]:
                best[0] = popcount(S)
            return
            
        # Pick a point v from candidates
        # Heuristic: pick the point with the most conflicts in S?
        # To keep it simple, just pick the lowest set bit.
        # Better: pick the one that blocks the most other candidates?
        
        # For now, just pick the first available
        lb = candidates & -candidates
        v = lb.bit_length() - 1
        candidates_without_v = candidates & ~lb
        
        # Option 1: Include v
        # Check if v can be added to S
        can_add = True
        blocked = 0
        temp_S = S
        while temp_S:
            lb_s = temp_S & -temp_S
            u = lb_s.bit_length() - 1
            temp_S -= lb_s
            k = partner[u][v]
            if k != -1:
                if S & (1 << k):
                    can_add = False
                    break
                else:
                    blocked |= (1 << k)
        
        if can_add:
            new_S = S | (1 << v)
            new_cands = candidates_without_v & ~blocked
            backtrack(new_S, new_cands)
            
        # Option 2: Exclude v
        backtrack(S, candidates_without_v)
        
    backtrack(0, (1 << n) - 1)
    return best[0]

def _generate_initial_config():
    """Generate a random valid config."""
    while True:
        points = []
        seen = set()
        # Try to generate a grid-like structure which has many collinearities
        # But random is safer for validity.
        for _ in range(500):
            points = []
            seen = set()
            while len(points) < 20:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                if (x, y) not in seen:
                    seen.add((x, y))
                    points.append([x, y])
            
            if _check_no_four_collinear(points):
                return points
        # Fallback
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        if _check_no_four_collinear(points):
            return points
        return None

def _local_search(points, max_iters=5000):
    """
    Local search to maximize the number of triples.
    """
    current_points = [list(p) for p in points]
    _, partner = _get_triples(current_points)
    current_triples_count = sum(1 for i in range(20) for j in range(i+1, 20) if partner[i][j] != -1)
    # Note: This counts pairs that are in a triple. Each triple has 3 pairs.
    # So triple_count = current_triples_count // 3
    
    best_points = [list(p) for p in current_points]
    best_triples_count = current_triples_count
    best_score = 21 # Initialize with high score
    
    # We will periodically evaluate the exact score
    last_eval_iter = 0
    
    for i in range(max_iters):
        if _elapsed() > _DEADLINE - 10:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        # Try a few random new positions
        found_improvement = False
        for _ in range(10):
            new_x = random.randint(old_x - 5, old_x + 5)
            new_y = random.randint(old_y - 5, old_y + 5)
            
            if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
                continue
            if (new_x, new_y) == (old_x, old_y):
                continue
            is_duplicate = False
            for j in range(20):
                if j != idx and current_points[j] == [new_x, new_y]:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
                
            new_points = [list(p) for p in current_points]
            new_points[idx] = [new_x, new_y]
            
            if not _check_no_four_collinear(new_points):
                continue
                
            _, new_partner = _get_triples(new_points)
            new_triples_count = sum(1 for i in range(20) for j in range(i+1, 20) if new_partner[i][j] != -1)
            
            if new_triples_count > current_triples_count:
                current_points = new_points
                partner = new_partner
                current_triples_count = new_triples_count
                if current_triples_count > best_triples_count:
                    best_triples_count = current_triples_count
                    best_points = [list(p) for p in current_points]
                found_improvement = True
                break
            elif new_triples_count == current_triples_count and random.random() < 0.1:
                current_points = new_points
                partner = new_partner
                current_triples_count = new_triples_count
                found_improvement = True
                break
                
        if not found_improvement and i % 100 == 0:
            idx = random.randint(0, 19)
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
            if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                is_dup = False
                for j in range(20):
                    if j != idx and current_points[j] == [new_x, new_y]:
                        is_dup = True
                        break
                if not is_dup:
                    new_points = [list(p) for p in current_points]
                    new_points[idx] = [new_x, new_y]
                    if _check_no_four_collinear(new_points):
                        current_points = new_points
                        _, partner = _get_triples(current_points)
                        current_triples_count = sum(1 for i in range(20) for j in range(i+1, 20) if partner[i][j] != -1)
                        
        # Periodic exact evaluation
        if i - last_eval_iter > 50:
            last_eval_iter = i
            score = _max_independent_set_size(partner, 20)
            if score < best_score:
                best_score = score
                best_points = [list(p) for p in current_points]
                if best_score <= 10:
                    return best_points

    # Final evaluation
    score = _max_independent_set_size(partner, 20)
    if score < best_score:
        best_score = score
        best_points = [list(p) for p in current_points]
        
    return best_points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_score = 21
    best_points = None
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if _elapsed() > _DEADLINE - 20:
            break
            
        init_points = _generate_initial_config()
        if not init_points:
            continue
            
        improved_points = _local_search(init_points, max_iters=2000)
        
        # Final check on the best from this restart
        _, partner = _get_triples(improved_points)
        score = _max_independent_set_size(partner, 20)
        
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in improved_points]
            if score <= 10:
                break
                
    if best_points is None:
        best_points = _generate_initial_config()
        
    return best_points

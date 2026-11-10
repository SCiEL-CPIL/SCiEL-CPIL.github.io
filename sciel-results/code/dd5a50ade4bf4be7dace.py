import random
import time
import sys
from itertools import combinations
from math import gcd

# Global deadline tracking
_START_TIME = None
DEADLINE_MARGIN = 15.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return 0.0
    return time.time() - _START_TIME

def _deadline_left():
    return 150.0 - DEADLINE_MARGIN - _elapsed()

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def count_collinear_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(p1, p2, points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def build_lines(points):
    """
    Returns a list of lines, where each line is a list of indices of points on that line.
    Only includes lines with >= 3 points.
    """
    n = len(points)
    line_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                continue
            
            # Normalize line equation ax + by + c = 0
            if dx == 0:
                a, b, c = 1, 0, -x1
            elif dy == 0:
                a, b, c = 0, 1, -y1
            else:
                g = gcd(abs(dx), abs(dy))
                dxn, dyn = dx // g, dy // g
                if dxn < 0:
                    dxn, dyn = -dxn, -dyn
                # a*x + b*y + c = 0
                # dyn*x - dxn*y + c = 0
                a, b = dyn, -dxn
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
            
            key = (a, b, c)
            if key not in line_dict:
                line_dict[key] = set()
            line_dict[key].add(i)
            line_dict[key].add(j)
    
    lines = []
    for pts in line_dict.values():
        if len(pts) >= 3:
            lines.append(list(pts))
    return lines

def max_independent_set_3_uniform(points):
    """
    Exact branch and bound for largest subset with no three collinear.
    Uses lines for pruning.
    """
    n = len(points)
    if n <= 2:
        return n

    lines = build_lines(points)
    if not lines:
        return n

    # Precompute for each point, the lines it is on
    point_lines = [[] for _ in range(n)]
    for idx, line in enumerate(lines):
        for p in line:
            point_lines[p].append(idx)

    # For each line, we can select at most 2 points.
    # We want to maximize selected points.
    # This is a packing problem.
    # B&B state:
    # - selected_mask: bitmask of selected points
    # - remaining_mask: bitmask of points not yet decided
    # - line_counts: count of selected points on each line
    
    # To make B&B fast, we need a good upper bound.
    # Upper bound: sum over all lines? No, lines overlap.
    # Simple bound: current_count + popcount(remaining_mask)
    # Better bound: current_count + sum over remaining points (1 if not blocked) ... complicated.
    # Let's stick to simple bound first, but order points by "degree" (number of lines they are on)
    # to prune earlier.
    
    # Actually, a simple greedy heuristic to find a good initial lower bound (best_size) is helpful.
    # Greedy: pick points one by one that allow max remaining.
    
    # Let's use the bitmask approach but with a better heuristic for point ordering.
    # Sort points by number of collinear triples they participate in (descending).
    # Points in more triples are more "constrained".
    
    triple_count_per_point = [0] * n
    for line in lines:
        # For each line, all pairs are in a triple with any other point on the line?
        # No, a triple is just 3 points.
        # If a point is on a line with k points, it is in C(k-1, 2) triples on that line.
        for p in line:
             k = len(line)
             triple_count_per_point[p] += (k-1)*(k-2)//2

    order = sorted(range(n), key=lambda i: -triple_count_per_point[i])
    
    # Map original index to position in order
    pos_in_order = [0] * n
    for pos, idx in enumerate(order):
        pos_in_order[idx] = pos

    best_size = 0
    full_mask = (1 << n) - 1
    
    # Precompute for each point i, the mask of points j such that i and j are in a line with >= 3 points?
    # No, we need to know if adding i creates a triple.
    # i creates a triple if there exist j, k in current_mask such that i, j, k are collinear.
    # This is equivalent to: for any line L containing i, if |L ∩ current_mask| >= 2, then i cannot be added.
    
    # Let's precompute line_masks: for each line, the bitmask of points on it.
    line_masks = [0] * len(lines)
    for idx, line in enumerate(lines):
        mask = 0
        for p in line:
            mask |= (1 << p)
        line_masks[idx] = mask

    def solve(pos, current_size, current_mask, line_counts):
        nonlocal best_size
        
        if _deadline_left() < 2:
            return
            
        # Upper bound: current_size + (n - pos)
        # This is weak.
        # Better: current_size + sum(1 for i in order[pos:] if not blocked)
        # Blocked means: for some line L containing i, line_counts[L] >= 2.
        # But line_counts changes.
        
        # Simple pruning:
        if current_size + (n - pos) <= best_size:
            return
            
        if pos == n:
            if current_size > best_size:
                best_size = current_size
            return

        p = order[pos]
        
        # Option 1: Exclude p
        solve(pos + 1, current_size, current_mask, line_counts)
        
        # Option 2: Include p
        # Check if p can be included
        can_include = True
        for l_idx in point_lines[p]:
            if line_counts[l_idx] >= 2:
                can_include = False
                break
        
        if can_include:
            new_mask = current_mask | (1 << p)
            new_counts = line_counts.copy()
            for l_idx in point_lines[p]:
                new_counts[l_idx] += 1
            
            solve(pos + 1, current_size + 1, new_mask, new_counts)

    # Initial lower bound via greedy
    # Greedy: iterate through order, if can add, add.
    greedy_mask = 0
    greedy_counts = [0] * len(lines)
    for p in order:
        can = True
        for l_idx in point_lines[p]:
            if greedy_counts[l_idx] >= 2:
                can = False
                break
        if can:
            greedy_mask |= (1 << p)
            for l_idx in point_lines[p]:
                greedy_counts[l_idx] += 1
    best_size = bin(greedy_mask).count('1')
    
    solve(0, 0, 0, [0] * len(lines))
    return best_size

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_score = 20  # Max possible
    best_points = None
    
    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))

    def evaluate(points):
        if has_four_collinear(points):
            return None
        return max_independent_set_3_uniform(points)

    # Strategy:
    # 1. Generate candidate configurations.
    #    - Grid-like structures (promote collinearity).
    #    - Random structures.
    # 2. For each, try to improve via local search (minimize score).
    # 3. Keep track of best.
    
    # Pre-defined good structures?
    # The "6 rows of 3" idea was flawed due to diagonals.
    # Let's try "5 rows of 4" but shifted to break diagonals.
    # Or just rely on random + local search with a good initialization.
    
    # Initialization:
    # Place points on a grid, then shuffle slightly.
    
    for restart in range(50):
        if _deadline_left() < 20: # Need time for eval
            break
            
        # Generate initial points
        # Try a few different distributions
        mode = restart % 3
        
        points = []
        seen = set()
        
        if mode == 0:
            # Grid
            ox, oy = random.randint(-50, 50), random.randint(-50, 50)
            sx, sy = random.randint(5, 15), random.randint(5, 15)
            for i in range(6):
                for j in range(4):
                    x = ox + i * sx + random.randint(-2, 2)
                    y = oy + j * sy + random.randint(-2, 2)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        if (x, y) not in seen:
                            seen.add((x, y))
                            points.append((x, y))
        elif mode == 1:
            # Clustered
            cx, cy = random.randint(-50, 50), random.randint(-50, 50)
            for _ in range(20):
                x = cx + random.randint(-20, 20)
                y = cy + random.randint(-20, 20)
                if -100 <= x <= 100 and -100 <= y <= 100:
                    if (x, y) not in seen:
                        seen.add((x, y))
                        points.append((x, y))
        else:
            # Random
            for _ in range(20):
                p = random_point()
                if p not in seen:
                    seen.add(p)
                    points.append(p)
        
        # Fill if necessary
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
                
        points = points[:20]
        
        # Ensure validity (no 4 collinear)
        valid = False
        attempts = 0
        while not valid and attempts < 50:
            if not has_four_collinear(points):
                valid = True
                break
            # Fix by moving a random point
            idx = random.randint(0, 19)
            old = points[idx]
            seen.discard(old)
            new = random_point()
            while new in seen:
                new = random_point()
            seen.add(new)
            points[idx] = new
            attempts += 1
            
        if not valid:
            continue
            
        # Evaluate
        score = evaluate(points)
        if score is None:
            continue
            
        if score < best_score:
            best_score = score
            best_points = points.copy()
            
        # Local Search
        current_points = points.copy()
        current_score = score
        
        for step in range(1000):
            if _deadline_left() < 10:
                break
                
            idx = random.randint(0, 19)
            old = current_points[idx]
            seen.discard(old)
            
            # Try a few new positions
            improved = False
            for _ in range(5):
                new = random_point()
                if new in seen:
                    continue
                    
                candidate = current_points.copy()
                candidate[idx] = new
                
                if has_four_collinear(candidate):
                    continue
                    
                # Quick check: count triples? Or just evaluate?
                # Evaluation is expensive.
                # Maybe use triple count as a proxy?
                # Lower triple count -> higher score? Not necessarily.
                # Let's just evaluate if triple count is promising.
                # Or just evaluate.
                
                new_score = evaluate(candidate)
                if new_score is None:
                    continue
                    
                if new_score < current_score:
                    current_points = candidate
                    current_score = new_score
                    seen.add(new)
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_points = current_points.copy()
                    break
                else:
                    # Accept with some probability? (Simulated Annealing)
                    # For now, greedy.
                    pass
            
            if improved:
                continue
                
            # If no improvement, maybe swap two points?
            # Or just break to save time?
            # Let's try swapping.
            if step % 10 == 0:
                idx2 = random.randint(0, 19)
                if idx2 == idx: continue
                p1, p2 = current_points[idx], current_points[idx2]
                seen.discard(p1); seen.discard(p2)
                
                new1, new2 = random_point(), random_point()
                if new1 in seen or new2 in seen or new1 == new2:
                    continue
                
                candidate = current_points.copy()
                candidate[idx] = new1
                candidate[idx2] = new2
                
                if has_four_collinear(candidate):
                    continue
                    
                new_score = evaluate(candidate)
                if new_score is not None and new_score < current_score:
                    current_points = candidate
                    current_score = new_score
                    seen.add(new1); seen.add(new2)
                    if current_score < best_score:
                        best_score = current_score
                        best_points = current_points.copy()

    if best_points is None:
        # Fallback
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        for _ in range(100):
            if not has_four_collinear(points):
                break
            idx = random.randint(0, 19)
            new = random_point()
            while new in seen:
                new = random_point()
            seen.add(new)
            points[idx] = new
        best_points = points

    return [list(p) for p in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)

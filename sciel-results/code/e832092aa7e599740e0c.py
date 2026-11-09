"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: ax + by + c = 0
    a = dy
    b = -dx
    c = -(a * x1 + b * y1)
    g2 = gcd(abs(a), abs(b))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    # Normalize sign
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _get_lines(points):
    """Return dict: line_key -> list of point indices on that line."""
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _has_four_collinear(points):
    lines = _get_lines(points)
    for pts in lines.values():
        if len(pts) >= 4:
            return True
    return False

def _count_triples(points):
    """Count number of collinear triples."""
    lines = _get_lines(points)
    count = 0
    for pts in lines.values():
        n = len(pts)
        if n >= 3:
            count += n * (n - 1) * (n - 2) // 6
    return count

def _max_3free_subset(points):
    """
    Find the largest subset of points with no three collinear using branch and bound.
    Uses bitmask representation and greedy matching lower bound for vertex cover.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Build list of collinear triples
    lines = _get_lines(points)
    triples = []
    for pts in lines.values():
        if len(pts) >= 3:
            idx_list = sorted(pts)
            for combo in combinations(idx_list, 3):
                triples.append(combo)
    
    if not triples:
        return n
    
    # For each point, which triples does it participate in?
    point_triples = [[] for _ in range(n)]
    for t in triples:
        for i in t:
            point_triples[i].append(t)
    
    # Precompute for each pair of points, the third point(s) that complete a triple
    pair_third = {}
    for t in triples:
        a, b, c = t
        pair_third.setdefault((a, b), set()).add(c)
        pair_third.setdefault((b, a), set()).add(c)
        pair_third.setdefault((a, c), set()).add(b)
        pair_third.setdefault((c, a), set()).add(b)
        pair_third.setdefault((b, c), set()).add(a)
        pair_third.setdefault((c, b), set()).add(a)
    
    best_size = [0]
    
    # Greedy lower bound on vertex cover to get upper bound on 3-free set
    # Actually, we want a good initial upper bound for the 3-free set
    # Greedy: repeatedly pick the point that is in the most remaining triples
    def greedy_3free():
        remaining = set(range(n))
        selected = []
        triples_set = set(triples)
        
        while remaining:
            # Find point in remaining that is in the most triples with other remaining points
            best_pt = None
            best_count = -1
            for pt in remaining:
                count = 0
                for t in point_triples[pt]:
                    # Check if all three are in remaining
                    if all(x in remaining for x in t):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_pt = pt
            
            if best_pt is None:
                break
            
            selected.append(best_pt)
            remaining.discard(best_pt)
            
            # Remove all triples that contain best_pt
            new_triples = []
            for t in triples_set:
                if best_pt not in t:
                    new_triples.append(t)
            triples_set = set(new_triples)
            
            if not triples_set:
                break
        
        return len(selected)
    
    # Initial upper bound from greedy
    greedy_bound = greedy_3free()
    best_size[0] = greedy_bound
    
    # Sort points by conflict degree (descending) for better pruning
    conflict_count = [len(point_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -conflict_count[i])
    
    def can_add(point, selected_mask):
        """Check if point can be added to selected set without creating a triple."""
        for i in range(n):
            if selected_mask & (1 << i):
                key = (i, point)
                if key in pair_third:
                    for k in pair_third[key]:
                        if selected_mask & (1 << k):
                            return False
        return True
    
    def dfs(idx, selected_mask, current_size):
        if _elapsed() > 115:
            return
        if idx == n:
            if current_size > best_size[0]:
                best_size[0] = current_size
            return
        
        point = order[idx]
        
        # Upper bound: current_size + number of remaining points
        if current_size + (n - idx) <= best_size[0]:
            return
        
        # Option 1: skip this point
        dfs(idx + 1, selected_mask, current_size)
        
        # Option 2: add this point if possible
        if can_add(point, selected_mask):
            dfs(idx + 1, selected_mask | (1 << point), current_size + 1)
    
    dfs(0, 0, 0)
    return best_size[0]

def _validate(points):
    """Check that points are valid: 20 distinct, |x|,|y|<=100, no four collinear."""
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
    if _has_four_collinear(points):
        return False
    return True

def _initial_grid():
    """Create a 4x5 grid with perturbations to avoid 4-collinear."""
    points = []
    for r in range(4):
        for c in range(5):
            x = c * 20 - 40
            y = r * 20 - 30
            px = x + (r % 2) * 3 + (c % 3) * 2
            py = y + (c % 2) * 3 + (r % 3) * 2
            points.append([px, py])
    return points

def _find_valid_initial():
    """Find a valid initial 20-point set with no four collinear."""
    # Try grid first
    pts = _initial_grid()
    if _validate(pts):
        return pts
    
    # Fallback: random search
    for _ in range(1000):
        points = []
        for _ in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = [x, y]
                if p not in points:
                    break
            points.append(p)
        if _validate(points):
            return points
    
    # Should not reach here
    return _initial_grid()

def _improve(points, deadline):
    """
    Local search to reduce the 3-free subset size.
    Strategy: move points to positions that increase collinear triple density.
    """
    current = [list(p) for p in points]
    current_score = _max_3free_subset(current)
    current_triples = _count_triples(current)
    
    best = [list(p) for p in current]
    best_score = current_score
    
    while _elapsed() < deadline:
        improved = False
        
        # Try moving each point
        for idx in range(20):
            if _elapsed() > deadline:
                break
            
            old_x, old_y = current[idx]
            best_new = None
            best_new_score = best_score
            best_new_triples = current_triples
            
            # Try various positions near the original point
            for dx in range(-15, 16, 3):
                for dy in range(-15, 16, 3):
                    if _elapsed() > deadline:
                        break
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 100 or abs(new_y) > 100:
                        continue
                    
                    new_p = [new_x, new_y]
                    if new_p in current:
                        continue
                    
                    new_points = current[:idx] + [new_p] + current[idx+1:]
                    if not _validate(new_points):
                        continue
                    
                    # Quick heuristic: count collinear triples
                    new_triples = _count_triples(new_points)
                    
                    if new_triples <= best_new_triples:
                        continue
                    
                    # Full evaluation
                    new_score = _max_3free_subset(new_points)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new = new_points[:]
                        best_new_triples = new_triples
            
            if best_new is not None:
                current = best_new
                current_triples = best_new_triples
                if best_new_score < best_score:
                    best_score = best_new_score
                    best = current[:]
                improved = True
        
        if not improved:
            # Try random restarts of local moves
            for _ in range(5):
                if _elapsed() > deadline:
                    break
                idx = random.randint(0, 19)
                old_x, old_y = current[idx]
                
                for _ in range(20):
                    if _elapsed() > deadline:
                        break
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    new_p = [new_x, new_y]
                    if new_p in current:
                        continue
                    
                    new_points = current[:idx] + [new_p] + current[idx+1:]
                    if not _validate(new_points):
                        continue
                    
                    new_triples = _count_triples(new_points)
                    if new_triples <= current_triples:
                        continue
                    
                    new_score = _max_3free_subset(new_points)
                    if new_score < best_score:
                        best_score = new_score
                        best = new_points[:]
                        current = new_points[:]
                        current_triples = new_triples
                        improved = True
                        break
            
            if not improved:
                break
    
    return best, best_score

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + 110  # Leave safety margin for 150s limit
    
    # Start with a valid initial configuration
    points = _find_valid_initial()
    best_points, best_score = _improve(points, deadline)
    
    # Try a few more random restarts if time permits
    while _elapsed() < deadline - 20:
        points = _find_valid_initial()
        candidate_points, candidate_score = _improve(points, min(deadline, _elapsed() + 30))
        
        if candidate_score < best_score:
            best_score = candidate_score
            best_points = candidate_points[:]
    
    return best_points
# EVOLVE-BLOCK-END

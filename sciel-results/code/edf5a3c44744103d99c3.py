"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
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
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = dy
    b = -dx
    c = -(a * x1 + b * y1)
    g2 = gcd(abs(a), abs(b))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
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
    """Find the largest subset of points with no three collinear using branch and bound."""
    n = len(points)
    lines = _get_lines(points)
    triples = []
    for pts in lines.values():
        if len(pts) >= 3:
            idx_list = sorted(pts)
            for combo in combinations(idx_list, 3):
                triples.append(combo)
    
    # Build conflict structure: for each point, which other points share a line with it
    # pair_third[i][j] = set of k such that i,j,k collinear
    pair_third = [set() for _ in range(n)]
    for t in triples:
        a, b, c = t
        pair_third[a].add((b, c))
        pair_third[b].add((a, c))
        pair_third[c].add((a, b))
    
    # For faster checking: for each point i, build a bitmask of "conflict pairs"
    # Actually, let's use a different approach:
    # For each point i, conflict_mask[i] = bitmask of points that form a triple with i and some other selected point
    # Better: for each pair (i,j), third[i][j] = bitmask of k collinear with i,j
    
    # Use a recursive B&B with good ordering
    # Sort by degree (number of triples each point is in)
    degree = [0] * n
    for t in triples:
        for i in t:
            degree[i] += 1
    order = sorted(range(n), key=lambda i: -degree[i])
    
    best_size = [0]
    
    # Precompute: for each point in order, what's its index
    # For checking if point p can be added to selected set S:
    #   for each pair (a,b) in S, check if p completes a triple
    #   equivalently: for each a in S, check if there exists b in S such that a,b,p collinear
    
    # Faster: for each point p, precompute lines_through_p[p] = list of lines through p
    # Then for each line through p, if 2+ points of that line are in S, p can't be added
    
    lines_through = [[] for _ in range(n)]
    for key, pts in lines.items():
        if len(pts) >= 3:
            for i in sorted(pts):
                lines_through[i].append(pts)
    
    def can_add_fast(p, selected_mask):
        """Check if point p can be added without creating a triple."""
        for line_pts in lines_through[p]:
            count_in_line = 0
            for i in line_pts:
                if i != p and (selected_mask >> i) & 1:
                    count_in_line += 1
                    if count_in_line >= 2:
                        return False
        return True
    
    def dfs(idx, selected_mask, current_size):
        if _elapsed() > 115:
            return
        if idx == n:
            if current_size > best_size[0]:
                best_size[0] = current_size
            return
        
        # Upper bound
        if current_size + (n - idx) <= best_size[0]:
            return
        
        point = order[idx]
        
        # Option 1: skip
        dfs(idx + 1, selected_mask, current_size)
        
        # Option 2: add if possible
        if can_add_fast(point, selected_mask):
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

def _build_candidate_pool():
    """Build a pool of candidate points for the QUBO-style selection."""
    pool = []
    # Dense grid in center
    for x in range(-50, 51, 2):
        for y in range(-50, 51, 2):
            pool.append((x, y))
    # Add some random points for diversity
    random.seed(123)
    for _ in range(500):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in pool:
            pool.append((x, y))
    return pool

def _compute_line_structure(pool):
    """For the candidate pool, compute which points are collinear with which others."""
    n = len(pool)
    # For each pair, find the line key
    # For each line, store the set of points on it
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pool[i], pool[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    # For each line, if it has >= 3 points, those points contribute to triples
    # Build: for each point i, list of lines through i that have >= 3 points
    lines_through = [[] for _ in range(n)]
    for key, pts in lines.items():
        if len(pts) >= 3:
            for i in pts:
                lines_through[i].append(pts)
    
    return lines, lines_through

def _qubo_simulated_annealing(pool, lines, lines_through, time_budget=100):
    """
    Simulated annealing to select 20 points maximizing collinear triples
    while avoiding 4-collinear.
    
    Energy = -num_triples + penalty * num_4collinear
    We want to minimize energy (maximize triples, avoid 4-collinear).
    """
    n_pool = len(pool)
    deadline = time.time() + time_budget
    
    # Start with a random valid configuration
    random.seed(42)
    
    def random_valid_config():
        """Generate a random configuration of 20 points with no 4-collinear."""
        for attempt in range(1000):
            pts = random.sample(range(n_pool), 20)
            # Check no 4-collinear
            valid = True
            for key, line_pts in lines.items():
                count = sum(1 for p in pts if p in line_pts)
                if count >= 4:
                    valid = False
                    break
            if valid:
                return pts
        return None
    
    def count_triples_config(pts):
        """Count collinear triples in a configuration of 20 point indices."""
        total = 0
        for key, line_pts in lines.items():
            count = sum(1 for p in pts if p in line_pts)
            if count >= 3:
                total += count * (count - 1) * (count - 2) // 6
        return total
    
    def has_four_collinear_config(pts):
        for key, line_pts in lines.items():
            count = sum(1 for p in pts if p in line_pts)
            if count >= 4:
                return True
        return False
    
    # Initialize
    current = random_valid_config()
    if current is None:
        # Fallback: use a structured initial
        current = list(range(20))
        while has_four_collinear_config(current):
            current = random.sample(range(n_pool), 20)
    
    current_triples = count_triples_config(current)
    best = list(current)
    best_triples = current_triples
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Generate neighbor: swap one point with another
        i = random.randint(0, 19)
        new_pt = random.randint(0, n_pool - 1)
        if new_pt in current:
            continue
        
        old_pt = current[i]
        new_config = list(current)
        new_config[i] = new_pt
        
        # Check validity (no 4-collinear)
        if has_four_collinear_config(new_config):
            continue
        
        new_triples = count_triples_config(new_config)
        delta = new_triples - current_triples  # We want to maximize, so positive delta is good
        
        if delta >= 0:
            current = new_config
            current_triples = new_triples
            if current_triples > best_triples:
                best_triples = current_triples
                best = list(current)
        else:
            # Accept with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                current = new_config
                current_triples = new_triples
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Occasionally do a 2-swap for larger moves
        if step % 50 == 0:
            i1, i2 = random.sample(range(20), 2)
            n1, n2 = random.sample(range(n_pool), 2)
            if n1 in current or n2 in current:
                continue
            new_config = list(current)
            new_config[i1] = n1
            new_config[i2] = n2
            if has_four_collinear_config(new_config):
                continue
            new_triples = count_triples_config(new_config)
            delta = new_triples - current_triples
            if delta >= 0 or random.random() < np.exp(delta / T):
                current = new_config
                current_triples = new_triples
                if current_triples > best_triples:
                    best_triples = current_triples
                    best = list(current)
    
    return best, best_triples

def _local_search_refine(points, time_budget=30):
    """Refine a configuration using local search to maximize triples."""
    deadline = time.time() + time_budget
    current = [list(p) for p in points]
    current_triples = _count_triples(current)
    best = [list(p) for p in points]
    best_triples = current_triples
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        old_x, old_y = current[idx]
        
        found_improvement = False
        for _ in range(50):
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
            if abs(new_x) > 100 or abs(new_y) > 100:
                continue
            
            # Check distinctness
            valid = True
            for j in range(20):
                if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                    valid = False
                    break
            if not valid:
                continue
            
            current[idx] = [new_x, new_y]
            
            if _has_four_collinear(current):
                current[idx] = [old_x, old_y]
                continue
            
            new_triples = _count_triples(current)
            if new_triples > current_triples:
                current_triples = new_triples
                if new_triples > best_triples:
                    best_triples = new_triples
                    best = [list(p) for p in current]
                found_improvement = True
                break
            else:
                current[idx] = [old_x, old_y]
        
        if not found_improvement:
            # Try larger perturbation
            for _ in range(20):
                new_x = old_x + random.randint(-20, 20)
                new_y = old_y + random.randint(-20, 20)
                if abs(new_x) > 100 or abs(new_y) > 100:
                    continue
                valid = True
                for j in range(20):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        valid = False
                        break
                if not valid:
                    continue
                current[idx] = [new_x, new_y]
                if _has_four_collinear(current):
                    current[idx] = [old_x, old_y]
                    continue
                new_triples = _count_triples(current)
                if new_triples > current_triples:
                    current_triples = new_triples
                    if new_triples > best_triples:
                        best_triples = new_triples
                        best = [list(p) for p in current]
                    break
                else:
                    current[idx] = [old_x, old_y]
    
    return best, best_triples

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    # Phase 1: Build candidate pool and compute line structure
    pool = _build_candidate_pool()
    n_pool = len(pool)
    
    # Compute line structure for the pool
    lines, lines_through = _compute_line_structure(pool)
    
    # Phase 2: QUBO-inspired simulated annealing to select 20 points
    sa_deadline = 80  # seconds
    selected_indices, sa_triples = _qubo_simulated_annealing(
        pool, lines, lines_through, time_budget=sa_deadline
    )
    
    # Convert to actual points
    points = [list(pool[i]) for i in selected_indices]
    
    # Validate
    if not _validate(points):
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
    
    # Phase 3: Local search refinement
    remaining_time = 110 - _elapsed()
    if remaining_time > 5:
        points, ls_triples = _local_search_refine(points, time_budget=max(5, remaining_time - 5))
        if _validate(points):
            pass  # Use refined points
        else:
            points = [list(pool[i]) for i in selected_indices]
    
    # Final validation
    if not _validate(points):
        return [[x, x * x] for x in range(-10, 10)]
    
    return points
# EVOLVE-BLOCK-END

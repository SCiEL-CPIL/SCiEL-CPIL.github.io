"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = math.gcd(abs(dx), abs(dy))
            if g == 0: continue
            dx //= g
            dy //= g
            count = 2
            for k in range(n):
                if k == i or k == j: continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _count_collinear_triples(points):
    """Count the number of collinear triples among the points."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Count points on the line
            count = 2
            for k in range(n):
                if k == i or k == j: continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            # Number of triples on this line is C(count, 3)
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
    return triples

def _max_3free_subset(points, deadline=None):
    """Branch and bound to find the largest subset with no three collinear."""
    n = len(points)
    best = [0]
    
    # Precompute collinearity: collinear[i][j][k] = True if i,j,k are collinear
    # Use a set of frozensets or a 3D boolean array? 
    # For n=20, 20^3 = 8000 booleans.
    collinear = [[[False]*n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                    collinear[i][j][k] = True
                    collinear[j][i][k] = True
                    collinear[k][i][j] = True
                    collinear[i][k][j] = True
                    collinear[j][k][i] = True
                    collinear[k][j][i] = True
    
    def dfs(idx, current, current_size):
        if deadline and time.time() > deadline:
            return
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pruning
        if current_size + (n - idx) <= best[0]:
            return
        
        # Try excluding idx
        dfs(idx + 1, current, current_size)
        
        # Try including idx
        can_add = True
        for i in current:
            for j in current:
                if i < j and collinear[idx][i][j]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            current.append(idx)
            dfs(idx + 1, current, current_size + 1)
            current.pop()
            
    dfs(0, [], 0)
    return best[0]

def _local_search(points, deadline):
    """
    Local search to maximize the number of collinear triples.
    Moves points to positions on lines defined by pairs of other points.
    """
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        # Pick a random point to move
        i = rng.randint(0, 19)
        orig = points[i][:]
        
        # Generate candidates: points on lines defined by pairs of other points
        candidates = set()
        for j in range(20):
            if j == i: continue
            for k in range(j+1, 20):
                if k == i: continue
                x1, y1 = points[j]
                x2, y2 = points[k]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                dx //= g
                dy //= g
                
                # Points on the line: (x1 + t*dx, y1 + t*dy)
                # Find integer t such that -100 <= x <= 100, -100 <= y <= 100
                # t range:
                if dx != 0:
                    t_min = max(math.ceil((-100 - x1) / dx), math.ceil((-100 - y1) / dy)) if dy != 0 else math.ceil((-100 - x1) / dx)
                    t_max = min(math.floor((100 - x1) / dx), math.floor((100 - y1) / dy)) if dy != 0 else math.floor((100 - x1) / dx)
                else:
                    # Vertical line
                    if -100 <= y1 <= 100 and -100 <= y2 <= 100:
                        for y in range(-100, 101):
                            if x1 == x2: # Check if x is valid
                                if -100 <= x1 <= 100:
                                    candidates.add((x1, y))
                    continue
                
                # Simplified: just iterate t in a reasonable range
                # Since coordinates are small, t is small
                for t in range(-200, 201):
                    x = x1 + t * dx
                    y = y1 + t * dy
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.add((x, y))
        
        # Also add some random nearby points
        for _ in range(10):
            nx = orig[0] + rng.randint(-5, 5)
            ny = orig[1] + rng.randint(-5, 5)
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                candidates.add((nx, ny))
        
        # Shuffle candidates
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        for cand in cand_list:
            # Check if cand is already used by another point
            if any(points[m] == list(cand) for m in range(20) if m != i):
                continue
            
            points[i] = list(cand)
            
            if _has_four_collinear(points):
                points[i] = orig
                continue
            
            new_triples = _count_collinear_triples(points)
            if new_triples > current_triples:
                current_triples = new_triples
                improved = True
                break
            else:
                points[i] = orig
        
        if not improved:
            # Try random restarts or accept worse moves (Simulated Annealing style)
            # For simplicity, just break if no improvement found in this iteration
            # But to avoid local optima, we can accept worse moves with some probability
            # Let's just break for now to keep it simple
            break
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_triples = -1
    
    # Try multiple random initializations
    for seed in range(100):
        if time.time() > deadline - 30:
            break
            
        rng = random.Random(seed)
        points = []
        used = set()
        
        # Generate 20 random points with no 4 collinear
        while len(points) < 20:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) in used:
                continue
            points.append([x, y])
            used.add((x, y))
            
            if _has_four_collinear(points):
                points.pop()
                used.remove((x, y))
                # To avoid infinite loop, if we fail many times, restart
                # But random points rarely have 4 collinear in small range? 
                # Actually, in 200x200, probability is low.
        
        if len(points) != 20:
            continue
            
        triples = _count_collinear_triples(points)
        
        # Local search
        improved_points = _local_search(points, deadline - 10)
        improved_triples = _count_collinear_triples(improved_points)
        
        if improved_triples > best_triples:
            best_triples = improved_triples
            best_points = [p[:] for p in improved_points]
            
    if best_points is not None:
        return best_points
    else:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END

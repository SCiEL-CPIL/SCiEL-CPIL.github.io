import time
import random
import math
from math import gcd

_T0 = time.time()
_DEADLINE = 135.0  # Safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a = dy
    b = -dx
    g = gcd(abs(a), abs(b))
    if g == 0:
        return None
    a //= g
    b //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * x1 + b * y1
    g2 = gcd(gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _count_exact4(points):
    """
    Returns (num_exact_4_lines, has_5_collinear).
    Optimized for speed.
    """
    n = len(points)
    if n != 16:
        return 0, True
    # Check for duplicates
    seen = set()
    for p in points:
        pt = tuple(p)
        if pt in seen:
            return 0, True
        seen.add(pt)
        
    pair_count = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            if k is None:
                continue
            pair_count[k] = pair_count.get(k, 0) + 1
    
    four = 0
    has5 = False
    for cnt in pair_count.values():
        if cnt == 6:
            four += 1
        elif cnt >= 10:
            has5 = True
            break
    return four, has5

def _score(points):
    """
    Score function for optimization.
    Higher is better.
    Penalizes 5-collinear and duplicates.
    """
    four, has5 = _count_exact4(points)
    if has5:
        return -1000 + four # Strong penalty, but keep some signal
    return four

def _random_points(rng):
    """Generate 16 random integer points within bounds."""
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        pt = (x, y)
        if pt not in seen:
            seen.add(pt)
            pts.append([x, y])
    return pts

def _perturb(points, rng, magnitude):
    """Perturb a few points by a random amount."""
    pts = [list(p) for p in points]
    num_moves = rng.randint(1, 3)
    for _ in range(num_moves):
        i = rng.randrange(16)
        ox, oy = pts[i]
        nx = ox + rng.randint(-magnitude, magnitude)
        ny = oy + rng.randint(-magnitude, magnitude)
        if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
            pts[i] = [nx, ny]
    return pts

def _simulated_annealing(initial_points, rng, time_limit):
    """
    Simulated Annealing with Move and Swap operations.
    """
    current_pts = [list(p) for p in initial_points]
    current_score = _score(current_pts)
    best_pts = [list(p) for p in current_pts]
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    start_time = time.time()
    while time.time() < start_time + time_limit:
        if T < T_min:
            break
            
        # Choose move type: 50% Move, 50% Swap
        if rng.random() < 0.5:
            # Move one point
            i = rng.randrange(16)
            new_pts = [list(p) for p in current_pts]
            ox, oy = new_pts[i]
            # Try a few random moves for this point
            for _ in range(5):
                nx = ox + rng.randint(-20, 20)
                ny = oy + rng.randint(-20, 20)
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    new_pts[i] = [nx, ny]
                    break
        else:
            # Swap two points
            i, j = rng.sample(range(16), 2)
            new_pts = [list(p) for p in current_pts]
            new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
            
        new_score = _score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = [list(p) for p in current_pts]
        else:
            # Accept with probability
            delta = current_score - new_score
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = math.exp(-delta / T)
            
            if rng.random() < accept_prob:
                current_pts = new_pts
                current_score = new_score
            
        T *= alpha
        
    return best_pts, best_score

def construct():
    t_start = time.time()
    rng = random.Random(42)
    
    # Baseline: 4x4 grid
    base = [[x, y] for x in range(4) for y in range(4)]
    best_pts = [list(p) for p in base]
    best_score = _score(best_pts)
    
    # Try multiple restarts with Simulated Annealing
    # Use different initial configurations
    initial_configs = [
        base,
        _random_points(rng),
        _random_points(rng),
        # A slightly perturbed grid
        [[x + rng.randint(-2, 2), y + rng.randint(-2, 2)] for x in range(4) for y in range(4)]
    ]
    
    # Filter out invalid initial configs (duplicates etc)
    valid_inits = []
    for cfg in initial_configs:
        s, has5 = _count_exact4(cfg)
        if not has5 and len(set(map(tuple, cfg))) == 16:
            valid_inits.append(cfg)
        else:
            # If invalid, try to fix or skip
            # For random points, duplicates are possible, so re-generate
            pass
            
    if not valid_inits:
        valid_inits = [base]
        
    # Allocate time for each run
    # Total budget ~ 130s
    # Leave 5s for final checks
    time_per_run = 25.0
    
    for i, init in enumerate(valid_inits):
        if time.time() > _DEADLINE - 5:
            break
            
        # Run SA
        cand_pts, cand_score = _simulated_annealing(init, rng, time_per_run)
        
        if cand_score > best_score:
            best_score = cand_score
            best_pts = [list(p) for p in cand_pts]
            
        # If we found a very high score (e.g., > 12), we might stop early?
        # But let's use the time.
        
    # Final local refinement of the best found
    if time.time() < _DEADLINE - 10:
        # Quick local search on best_pts
        for _ in range(100):
            if time.time() > _DEADLINE - 5:
                break
            i = rng.randrange(16)
            ox, oy = best_pts[i]
            for _ in range(10):
                nx = ox + rng.randint(-5, 5)
                ny = oy + rng.randint(-5, 5)
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    old = best_pts[i]
                    best_pts[i] = [nx, ny]
                    s, has5 = _count_exact4(best_pts)
                    if has5:
                        s = -1000
                    if s > best_score:
                        best_score = s
                    else:
                        best_pts[i] = old
                        
    # Final validation
    final_pts = [list(p) for p in best_pts]
    final_score, has5 = _count_exact4(final_pts)
    
    if has5 or len(set(tuple(p) for p in final_pts)) != 16:
        # Fallback to grid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return final_pts

if __name__ == "__main__":
    r = construct()
    print(len(r), r[:4])
    print(_count_exact4(r))

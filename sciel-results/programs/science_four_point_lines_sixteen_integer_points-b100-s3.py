import time
import random
import math
from collections import defaultdict

def _canonical_line(p1, p2):
    """
    Returns a canonical representation of the line passing through p1 and p2.
    p1, p2 are tuples (x, y).
    The line is defined by ax + by + c = 0.
    We normalize (a, b, c) such that:
    1. gcd(|a|, |b|, |c|) = 1
    2. a > 0, or a == 0 and b > 0, or a == 0 and b == 0 and c > 0
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    
    # Handle vertical/horizontal/special cases implicitly via gcd
    # If a and b are both 0, it's not a line, but points are distinct so this won't happen for distinct pairs.
    
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    
    # Normalization for sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def _score(points):
    """
    Calculate the score for a set of 16 points.
    Maximize lines with exactly 4 points.
    Penalize lines with >= 5 points.
    """
    line_counts = defaultdict(int)
    n = len(points)
    # Count points on each line defined by pairs
    # To avoid counting the same line multiple times in the dict keys, 
    # we just map (line_key) -> set of points? No, that's heavy.
    # Better: Map (line_key) -> count.
    # But we must be careful: if we iterate pairs, we add 1 for each pair?
    # No, we want to know how many points are on the line.
    # Efficient way: 
    # For each unique line key, we need to know the number of points on it.
    # We can just use a dict: key -> count.
    # But how to count?
    # If we iterate all pairs (i, j), we get the line L. 
    # We can't just increment a counter for L because multiple pairs on the same line will increment it.
    # We need to count distinct points.
    
    # With N=16, N^2 = 256 pairs.
    # We can store: line_key -> set of point indices.
    # Then at the end, score = sum(1 for s in sets if len(s)==4) - 100 * sum(1 for s in sets if len(s)>=5)
    
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            key = _canonical_line(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
            
    score = 0
    for pts in line_points.values():
        if len(pts) == 4:
            score += 10
        elif len(pts) >= 5:
            score -= 100
            
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial random configuration
    # Use a box size that allows enough room for distinct lines
    # 1000 is the limit. Let's use [-50, 50] initially to keep numbers small for speed, 
    # but the problem allows up to 1000. 
    # Actually, smaller numbers might lead to more accidental collinearities?
    # Or maybe larger numbers are better to avoid "small integer" patterns?
    # Let's try [-20, 20] first.
    
    def random_point():
        return [random.randint(-20, 20), random.randint(-20, 20)]
        
    points = []
    while len(points) < 16:
        p = random_point()
        if p not in points:
            points.append(p)
            
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    # Simulated Annealing
    T = 50.0
    min_T = 0.1
    cooling = 0.99
    
    while time.time() < deadline:
        # Move a random point
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # New position: random nearby or random in box?
        # Random in box is better for global exploration.
        # Let's mix: 50% nearby, 50% global.
        if random.random() < 0.5:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
            
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_point = [new_x, new_y]
        
        # Check distinctness
        if any(new_point == p for i, p in enumerate(points) if i != idx):
            continue
            
        # Apply move
        points[idx] = new_point
        
        new_score = _score(points)
        
        if new_score >= best_score:
            best_score = new_score
            best_points = [p[:] for p in points]
        else:
            # Accept worse solution with probability exp(delta/T)
            delta = new_score - best_score
            if random.random() < math.exp(delta / T):
                # Accept, but don't update best
                pass
            else:
                # Reject move
                points[idx] = [old_x, old_y]
                
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_points

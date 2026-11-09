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
    
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def _compute_line_map(points):
    """Compute line -> set of point indices mapping."""
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _canonical_line(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _score(points):
    """
    Calculate the score for a set of 16 points.
    Returns (num_4_lines, has_5_plus) where:
    - num_4_lines: count of lines with exactly 4 points
    - has_5_plus: True if any line has >= 5 points (invalid)
    """
    line_points = _compute_line_map(points)
    num_4 = 0
    has_5 = False
    for pts in line_points.values():
        if len(pts) == 4:
            num_4 += 1
        elif len(pts) >= 5:
            has_5 = True
    return num_4, has_5

def _anneal_score(points):
    """
    Score for annealing: maximize 4-lines, heavily penalize 5+ lines.
    """
    num_4, has_5 = _score(points)
    if has_5:
        return num_4 - 1000
    return num_4

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    BOX = 100
    
    def random_point_in_box(box):
        return [random.randint(-box, box), random.randint(-box, box)]
        
    # Initial random configuration in a moderate box
    points = []
    while len(points) < 16:
        p = random_point_in_box(BOX)
        if p not in points:
            points.append(p)
            
    best_points = [p[:] for p in points]
    best_num4, best_has5 = _score(best_points)
    best_ann_score = best_num4 - (1000 if best_has5 else 0)
    
    # Simulated Annealing
    T = 10.0
    min_T = 0.05
    cooling = 0.995
    
    while time.time() < deadline:
        # Move a random point
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # New position: mix of local, medium, and global moves
        r = random.random()
        if r < 0.3:
            # Local: small perturbation
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
        elif r < 0.7:
            # Medium: moderate jump
            new_x = old_x + random.randint(-20, 20)
            new_y = old_y + random.randint(-20, 20)
        else:
            # Global: random in full box
            new_x = random.randint(-BOX, BOX)
            new_y = random.randint(-BOX, BOX)
            
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_point = [new_x, new_y]
        
        # Check distinctness
        if any(new_point == p for i, p in enumerate(points) if i != idx):
            continue
            
        # Apply move
        points[idx] = new_point
        
        num4, has5 = _score(points)
        ann_score = num4 - (1000 if has5 else 0)
        
        if ann_score > best_ann_score:
            best_ann_score = ann_score
            best_points = [p[:] for p in points]
            best_num4 = num4
            best_has5 = has5
        else:
            # Accept worse solution with probability exp(delta/T)
            delta = ann_score - best_ann_score
            if delta < -500:  # Avoid overflow
                accept = False
            else:
                accept = random.random() < math.exp(delta / T)
            if accept:
                pass
            else:
                # Reject move
                points[idx] = [old_x, old_y]
                
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Final polish: try to improve best_points with focused local search
    if time.time() < deadline - 5:
        polish_points = [p[:] for p in best_points]
        polish_num4, polish_has5 = _score(polish_points)
        polish_ann = polish_num4 - (1000 if polish_has5 else 0)
        
        T2 = 1.0
        for _ in range(2000):
            if time.time() > deadline - 3:
                break
            idx = random.randint(0, 15)
            old_x, old_y = polish_points[idx]
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            np = [new_x, new_y]
            if any(np == p for i, p in enumerate(polish_points) if i != idx):
                continue
            polish_points[idx] = np
            n4, h5 = _score(polish_points)
            pa = n4 - (1000 if h5 else 0)
            if pa > polish_ann:
                polish_ann = pa
                polish_num4 = n4
                polish_has5 = h5
            else:
                delta = pa - polish_ann
                if delta < -100:
                    accept = False
                else:
                    accept = random.random() < math.exp(delta / T2)
                if not accept:
                    polish_points[idx] = [old_x, old_y]
            T2 *= 0.999
            if T2 < 0.01:
                T2 = 0.01
        
        if polish_ann > best_ann_score:
            best_points = polish_points
    
    # Validate: ensure no 5-collinear in returned points
    final_num4, final_has5 = _score(best_points)
    if final_has5:
        # Fallback: generate a valid random config
        fallback = []
        while len(fallback) < 16:
            p = random_point_in_box(100)
            if p not in fallback:
                fallback.append(p)
        fn4, fh5 = _score(fallback)
        if not fh5:
            best_points = fallback
        else:
            # Last resort: use a known-valid configuration
            # 4x4 grid has no 5-collinear but few 4-lines
            best_points = [[i, j] for i in range(4) for j in range(4)]
    
    return best_points

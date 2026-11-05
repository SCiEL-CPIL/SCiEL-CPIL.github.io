"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _doubled_area(p1, p2, p3):
    """Compute absolute doubled area of triangle with vertices p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return abs(det)

def _full_score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _make_grid():
    """Create a 6x5 grid as initial configuration."""
    return [(x, y) for x in range(6) for y in range(5)]

def _valid(points):
    """Check if all points are distinct and in [-30,30]^2."""
    if len(points) != 30:
        return False
    seen = set()
    for x, y in points:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _try_move(points, idx, new_pos):
    """Try moving point at idx to new_pos, return new score and new points if valid."""
    if new_pos in points:
        return None, None
    new_points = list(points)
    new_points[idx] = new_pos
    if not _valid(new_points):
        return None, None
    score = _full_score(new_points)
    return score, new_points

def _local_search(points, deadline, rng):
    """Perform local hill-climbing search."""
    current_points = list(points)
    current_score = _full_score(current_points)
    best_points = list(current_points)
    best_score = current_score
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if no_improve >= max_no_improve:
            break
        # Try moving a random point to a random position
        idx = rng.randint(0, 29)
        x = rng.randint(-30, 30)
        y = rng.randint(-30, 30)
        new_pos = (x, y)
        
        new_score, new_points = _try_move(current_points, idx, new_pos)
        if new_score is None:
            no_improve += 1
            continue
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
        else:
            no_improve += 1
    
    return best_points, best_score

def construct():
    deadline = time.time() + 85  # Leave 5s margin before 90s limit
    rng = random.Random(42)
    
    # Start with grid
    initial = _make_grid()
    best_points, best_score = _local_search(initial, deadline, rng)
    
    # Try a few more restarts with perturbed starts if time allows
    while time.time() < deadline - 5:
        # Perturb the best solution
        perturbed = list(best_points)
        for _ in range(3):
            idx = rng.randint(0, 29)
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in perturbed:
                perturbed[idx] = (x, y)
        
        if not _valid(perturbed):
            continue
        
        p_points, p_score = _local_search(perturbed, deadline, rng)
        if p_score > best_score:
            best_score = p_score
            best_points = p_points
    
    return best_points
# EVOLVE-BLOCK-END

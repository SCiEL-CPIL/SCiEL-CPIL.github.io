"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations
from fractions import Fraction

# EVOLVE-BLOCK-START
def _get_line_key(p1, p2):
    """Get a canonical representation of the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Canonical form: ensure dx > 0, or dx == 0 and dy > 0
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Normalized line: (dx, dy, c) where c = dy*x1 - dx*y1 (adjusted)
    # Line equation: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_lines_with_exactly_four(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    
    # Count points on each line
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
    
    # Count lines with exactly 4 points
    count = 0
    for key, indices in line_counts.items():
        if len(indices) == 4:
            count += 1
    return count

def _has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    if n < 5:
        return False
    
    # Count points on each line
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
    
    for key, indices in line_counts.items():
        if len(indices) >= 5:
            return True
    return False

def _score(points):
    """Score the configuration: number of lines with exactly 4 points, 0 if invalid."""
    if _has_five_collinear(points):
        return 0
    return _count_lines_with_exactly_four(points)

def _initial_grid():
    """Create a 4x4 grid centered at origin."""
    return [[x - 1.5, y - 1.5] for x in range(4) for y in range(4)]

def _random_point():
    """Generate a random integer point within bounds."""
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _perturb(points, intensity=10):
    """Perturb one random point by a random amount."""
    new_points = [p[:] for p in points]
    idx = random.randint(0, len(new_points) - 1)
    new_points[idx][0] += random.randint(-intensity, intensity)
    new_points[idx][1] += random.randint(-intensity, intensity)
    # Clamp to bounds
    new_points[idx][0] = max(-1000, min(1000, new_points[idx][0]))
    new_points[idx][1] = max(-1000, min(1000, new_points[idx][1]))
    return new_points

def construct():
    """Find 16 integer points maximizing lines with exactly 4 points."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(best_points)
    
    # Simulated annealing
    temperature = 50.0
    cooling = 0.995
    min_temp = 0.1
    
    while time.time() < deadline:
        # Perturb
        new_points = _perturb(best_points, intensity=15)
        
        # Check if all points are distinct and within bounds
        valid = True
        seen = set()
        for p in new_points:
            if p[0] < -1000 or p[0] > 1000 or p[1] < -1000 or p[1] > 1000:
                valid = False
                break
            key = (p[0], p[1])
            if key in seen:
                valid = False
                break
            seen.add(key)
        
        if not valid:
            continue
        
        new_score = _score(new_points)
        
        if new_score >= best_score:
            best_points = new_points
            best_score = new_score
        else:
            # Accept with probability exp((new_score - best_score) / temperature)
            delta = new_score - best_score
            if delta < 0:
                prob = math.exp(delta / temperature) if temperature > 0 else 0
                if random.random() < prob:
                    best_points = new_points
                    best_score = new_score
        
        # Cool down
        temperature = max(min_temp, temperature * cooling)
        
        # Occasionally restart with a fresh perturbation
        if random.random() < 0.01:
            temperature = 30.0
    
    return best_points
# EVOLVE-BLOCK-END


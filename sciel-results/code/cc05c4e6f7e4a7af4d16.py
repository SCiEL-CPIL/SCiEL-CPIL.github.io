"""Search-based program for maximizing 4-point lines among 16 integer points."""
import time
import random
import math
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0
_DEADLINE_EFF = _START_TIME + _DEADLINE - _SAFETY


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _canonical_line(p1, p2):
    """Return a canonical representation of the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Canonical direction: make dx positive, or if dx==0 make dy positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line equation: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    return (dx, dy, c)


def _count_4lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    line_to_count = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            if line is None:
                continue
            line_to_count[line] = line_to_count.get(line, 0) + 1
    
    four_count = 0
    for line, count in line_to_count.items():
        if count == 6:  # C(4,2) = 6 pairs for 4 points
            four_count += 1
    return four_count


def _has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    if n < 5:
        return False
    line_to_count = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            if line is None:
                continue
            line_to_count[line] = line_to_count.get(line, 0) + 1
    for line, count in line_to_count.items():
        if count >= 10:  # C(5,2) = 10 pairs for 5 points
            return True
    return False


def _valid(points):
    """Check if the configuration is valid: 16 distinct points, |x|,|y|<=1000, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if _has_five_collinear(points):
        return False
    return True


def _score(points):
    """Return the score (number of 4-point lines) if valid, else -1."""
    if not _valid(points):
        return -1
    return _count_4lines(points)


def _initial_grid():
    """Return the initial 4x4 grid."""
    return [[x, y] for x in range(4) for y in range(4)]


def _random_point():
    """Generate a random integer point within bounds."""
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]


def _perturb(points, num_changes=1):
    """Create a perturbed version of the points."""
    new_points = [list(p) for p in points]
    indices = random.sample(range(16), min(num_changes, 16))
    for idx in indices:
        new_points[idx] = _random_point()
    return new_points


def _local_search(points, max_iter=1000):
    """Perform local search to improve the score."""
    current_score = _score(points)
    if current_score < 0:
        return points, current_score
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE_EFF:
            break
        
        # Try changing one point at a time
        best_neighbor = None
        best_neighbor_score = current_score
        
        for i in range(16):
            if time.time() > _DEADLINE_EFF:
                break
            original = points[i]
            # Try a few random replacements
            for _ in range(5):
                candidate = [list(p) for p in points]
                candidate[i] = _random_point()
                s = _score(candidate)
                if s > best_neighbor_score:
                    best_neighbor_score = s
                    best_neighbor = candidate
        
        if best_neighbor is not None:
            points = best_neighbor
            current_score = best_neighbor_score
        else:
            # No improvement found, try a larger perturbation
            points = _perturb(points, num_changes=2)
            current_score = _score(points)
            if current_score < 0:
                break
    
    return points, current_score


def construct():
    """Main function: search for a configuration with many 4-point lines."""
    best_points = _initial_grid()
    best_score = _score(best_points)
    
    # Try multiple random restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > _DEADLINE_EFF:
            break
        
        if restart == 0:
            points = _initial_grid()
        else:
            # Generate a random starting configuration
            points = [_random_point() for _ in range(16)]
        
        points, score = _local_search(points, max_iter=200)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    # Final verification
    if not _valid(best_points):
        # Fall back to the grid
        best_points = _initial_grid()
        best_score = _score(best_points)
    
    return [list(p) for p in best_points]

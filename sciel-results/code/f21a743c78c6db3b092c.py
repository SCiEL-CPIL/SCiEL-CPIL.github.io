import time
import random
from itertools import combinations

def _collinear(p1, p2, p3):
    """Check if three points are collinear."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    for l in range(k+1, n):
                        if _collinear(points[i], points[j], points[l]):
                            return True
    return False

def _count_triples(points):
    """Count number of collinear triples."""
    count = 0
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_3free_subset(points):
    """Find the largest subset with no three collinear using branch and bound."""
    n = len(points)
    
    # Precompute for each pair whether adding a third point creates a collinear triple
    # collinear[i][j] = set of k such that i,j,k are collinear
    collinear_with = [[set() for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(n):
                if k != i and k != j and _collinear(points[i], points[j], points[k]):
                    collinear_with[i][j].add(k)
                    collinear_with[j][i].add(k)
    
    best = [0]
    
    def branch(remaining, current_size, used_points):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(remaining) <= best[0]:
            return
        
        # Pick a point to try
        idx = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include idx
        # Check if including idx is valid: no two points in used_points form a collinear triple with idx
        valid = True
        for i, j in combinations(used_points, 2):
            if idx in collinear_with[i][j]:
                valid = False
                break
        if valid:
            branch(rest, current_size + 1, used_points + [idx])
        
        # Option 2: exclude idx
        branch(rest, current_size, used_points)
    
    branch(list(range(n)), 0, [])
    return best[0]

def _validate(points):
    """Validate: 20 distinct points, |x|,|y| <= 100, no four collinear."""
    if len(points) != 20:
        return False
    if len(set(tuple(p) for p in points)) != 20:
        return False
    for p in points:
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            return False
    if _has_four_collinear(points):
        return False
    return True

def _score(points):
    """Compute the score (max 3-free subset size)."""
    return _max_3free_subset(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with a grid-based construction
    # Try to create many 3-point lines
    
    # Initial guess: use a perturbed grid
    # Let's try several random configurations and keep the best
    
    best_points = None
    best_score = 20  # worst case
    
    # Try a structured construction first
    # Place points to create many 3-collinear triples
    
    # Construction: use a 5x4 grid but perturb to avoid 4-collinear
    # Actually, let's try specific known-good patterns
    
    # Try: points on multiple lines
    # Line 1: (0,0), (1,0), (2,0) - but this is a line with 3
    # We need to be careful not to create 4-collinear
    
    # Let me try a different approach: random search with local optimization
    
    random.seed(42)
    
    # Generate initial candidate
    def generate_candidate():
        """Generate a random candidate configuration."""
        points = []
        used = set()
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in used:
                points.append([x, y])
                used.add((x, y))
        return points
    
    # Try several random starts
    for attempt in range(5):
        if time.time() > deadline:
            break
        points = generate_candidate()
        if not _validate(points):
            continue
        s = _score(points)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in points]
    
    # Local search: try to improve by moving points
    if best_points is not None:
        current = [list(p) for p in best_points]
        current_score = best_score
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Try moving one point to a random position
            improved = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = random.randint(0, 19)
                old_x, old_y = current[i]
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
                current[i] = [new_x, new_y]
                
                if _validate(current):
                    s = _score(current)
                    if s < current_score:
                        current_score = s
                        best_score = s
                        best_points = [list(p) for p in current]
                        improved = True
                    else:
                        current[i] = [old_x, old_y]
                else:
                    current[i] = [old_x, old_y]
            
            if not improved:
                break
    
    if best_points is None:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points

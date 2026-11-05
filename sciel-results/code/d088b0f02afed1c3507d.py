"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from math import gcd
import random

def count_two_point_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    two_point_lines = sum(1 for pts in line_to_points.values() if len(pts) == 2)
    return two_point_lines

def all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
            return False
    return True

def construct():
    """
    Construct 12 points minimizing the number of 2-point lines.
    Uses Simulated Annealing with line-based move generation.
    """
    
    N = 12
    random.seed(42)
    
    # Initial configuration: 3x4 grid
    # Scale it up to allow for more "room" for moves if needed, 
    # but small integers are fine for counting.
    # Let's start with a 3x4 grid centered at 0.
    initial_points = []
    for x in range(4):
        for y in range(3):
            initial_points.append([x - 1.5, y - 1]) # Wait, must be integers.
            # Let's use integers: x in -1,0,1,2? No, 0,1,2,3 is fine.
    initial_points = [[x, y] for x in range(4) for y in range(3)]
    
    # Another candidate: 2x6 grid
    initial_points2 = [[x, y] for x in range(6) for y in range(2)]
    
    # Another candidate: Random
    def random_points():
        pts = []
        while len(pts) < N:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if p not in pts:
                pts.append(p)
        if not all_collinear(pts):
            return pts
        return random_points()
    
    candidates = [initial_points, initial_points2, random_points()]
    
    best_score = float('inf')
    best_config = None
    
    # Simulated Annealing parameters
    T_start = 10.0
    T_end = 0.001
    alpha = 0.99
    max_iters = 2000
    
    def anneal(points):
        points = [tuple(p) for p in points]
        current_score = count_two_point_lines(points)
        current_points = points[:]
        T = T_start
        
        for it in range(max_iters):
            T *= alpha
            if T < T_end:
                break
            
            # Choose a random point to move
            i = random.randint(0, N-1)
            old_point = current_points[i]
            
            # Generate candidate moves
            # 1. Move to a point on a line defined by two other points
            # 2. Small random jump
            
            cands = set()
            
            # Line-based moves
            # For each pair of other points (j, k), find integer points on their line
            # near the current point or within a reasonable range
            for j in range(N):
                if i == j: continue
                for k in range(j+1, N):
                    if i == k: continue
                    x1, y1 = current_points[j]
                    x2, y2 = current_points[k]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    g = gcd(abs(dx), abs(dy))
                    if g == 0: continue # Should not happen for distinct points
                    
                    sx = dx // g
                    sy = dy // g
                    
                    # Parameterize line: P = P1 + t * (sx, sy)
                    # We want points close to old_point or just valid integer points.
                    # To be efficient, we check t values that put the point in a box around old_point
                    # or just a small set of t values.
                    # Let's try t in [-5, 5] relative to P1?
                    # Better: Calculate t such that P is close to old_point.
                    # Projection of (old - P1) onto (sx, sy)
                    # dot = (ox - x1)*sx + (oy - y1)*sy
                    # len2 = sx^2 + sy^2
                    # t_proj = dot / len2
                    ox, oy = old_point
                    dot = (ox - x1)*sx + (oy - y1)*sy
                    len2 = sx*sx + sy*sy
                    t_center = dot / len2
                    
                    # Check integer t values around t_center
                    for dt in range(-5, 6):
                        t = int(t_center) + dt
                        nx = x1 + t * sx
                        ny = y1 + t * sy
                        if -10**9 <= nx <= 10**9 and -10**9 <= ny <= 10**9:
                            cands.add((nx, ny))
            
            # Random moves
            for _ in range(5):
                nx = old_point[0] + random.randint(-3, 3)
                ny = old_point[1] + random.randint(-3, 3)
                cands.add((nx, ny))
            
            # Filter candidates: must be distinct from other points
            valid_cands = []
            other_points = set(current_points)
            other_points.discard(old_point)
            
            for p in cands:
                if p in other_points:
                    continue
                # Check bounds
                if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                    continue
                valid_cands.append(p)
            
            if not valid_cands:
                continue
                
            new_point = random.choice(valid_cands)
            
            # Apply move
            test_points = current_points[:]
            test_points[i] = new_point
            
            # Check distinctness (already done) and collinearity
            if all_collinear(test_points):
                continue
                
            new_score = count_two_point_lines(test_points)
            
            # Acceptance criterion
            if new_score < current_score or random.random() < pow(2.718, (current_score - new_score) / T):
                current_points = test_points
                current_score = new_score
                
        return [list(p) for p in current_points], current_score

    for config in candidates:
        improved_config, score = anneal(config)
        if score < best_score:
            best_score = score
            best_config = improved_config

    if best_config is None:
        best_config = [[x, y] for x in range(4) for y in range(3)]
        
    return best_config

# EVOLVE-BLOCK-END

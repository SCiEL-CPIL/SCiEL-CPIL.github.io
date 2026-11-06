import random
from itertools import combinations

def count_ordinary_lines(points):
    """Count lines containing exactly 2 points."""
    n = len(points)
    # Use a set of normalized line representations
    line_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize the direction vector
            if dx == 0:
                # Vertical line
                key = ('v', x1)
            elif dy == 0:
                # Horizontal line
                key = ('h', y1)
            else:
                # General line: normalize (dx, dy) to primitive direction
                from math import gcd
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                # Ensure consistent sign
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                # Line is determined by point and direction: use intercept form
                # y - y1 = m(x - x1) => m*x - y = m*x1 - y1
                # Use (dy_n, -dx_n, dy_n*x1 - dx_n*y1) as line key, normalized
                a = dy_n
                b = -dx_n
                c = a * x1 + b * y1
                # Normalize (a, b, c) so that gcd(a,b,c)=1 and a>0 or (a==0 and b>0)
                g2 = gcd(abs(a), abs(b))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                key = ('g', a, b, c)
            
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    """Find 12 integer points minimizing ordinary lines via local search."""
    
    # Start with a good initial configuration
    # Try a 3x4 grid first
    initial = [[x, y] for x in range(3) for y in range(4)]
    best_score = count_ordinary_lines(initial)
    best_points = [p[:] for p in initial]
    
    # Also try other initial configurations
    # 4x3 grid
    initial2 = [[x, y] for x in range(4) for y in range(3)]
    score2 = count_ordinary_lines(initial2)
    if score2 < best_score:
        best_score = score2
        best_points = [p[:] for p in initial2]
    
    # Try a configuration with a line of 4 and others
    # e.g., 4 points on y=0, 4 on y=1, 4 on y=2 with shifted x
    initial3 = []
    for y in range(3):
        for x in range(4):
            initial3.append([x + y, y])  # shifted
    score3 = count_ordinary_lines(initial3)
    if score3 < best_score:
        best_score = score3
        best_points = [p[:] for p in initial3]
    
    # Local search: try moving individual points
    random.seed(42)
    max_iterations = 5000
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    for iteration in range(max_iterations):
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Try several candidate positions
        improved = False
        for _ in range(20):
            # Random new position
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            
            # Skip if same position
            if new_x == old_x and new_y == old_y:
                continue
            
            # Skip if duplicates with other points
            dup = False
            for j in range(12):
                if j != idx and current_points[j][0] == new_x and current_points[j][1] == new_y:
                    dup = True
                    break
            if dup:
                continue
            
            current_points[idx] = [new_x, new_y]
            new_score = count_ordinary_lines(current_points)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                break
            else:
                current_points[idx] = [old_x, old_y]
        
        if not improved:
            # Try swapping two points
            i, j = random.sample(range(12), 2)
            tmp = current_points[i][:]
            current_points[i] = current_points[j][:]
            current_points[j] = tmp
            
            # Check for duplicates
            dup = False
            seen = set()
            for p in current_points:
                if tuple(p) in seen:
                    dup = True
                    break
                seen.add(tuple(p))
            
            if not dup:
                new_score = count_ordinary_lines(current_points)
                if new_score < current_score:
                    current_score = new_score
                else:
                    # Revert
                    current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
        
        # Early stop if we reach theoretical minimum
        if best_score <= 6:
            break
    
    return best_points

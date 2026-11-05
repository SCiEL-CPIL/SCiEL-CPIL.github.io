import time
import random
import math

def _score(points):
    """
    Calculate the number of lines containing exactly 4 points.
    Also returns 0 if any line has 5 or more points (invalid).
    Returns -1 if points are not distinct.
    """
    n = len(points)
    # Check distinctness
    if len(set(tuple(p) for p in points)) != n:
        return -1
        
    # Map from line key to set of points on that line
    # Line key: (a, b, c) for ax + by + c = 0, normalized
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line equation: ax + by + c = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # Normalize the line equation
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                continue # Should not happen for distinct points
                
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    four_lines = 0
    for pts in line_points.values():
        k = len(pts)
        if k == 4:
            four_lines += 1
        elif k >= 5:
            return 0 # Invalid configuration: 5 or more collinear
            
    return four_lines

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Initial configuration: 4x4 grid
    # This configuration has 10 lines with exactly 4 points:
    # 4 rows, 4 columns, 2 main diagonals.
    points = [[x, y] for x in range(4) for y in range(4)]
    
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    n = 16
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Pre-allocate lists for speed if needed, but n=16 is small
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = points[idx]
        
        # Generate a new position
        # Mix of small steps and larger jumps
        if random.random() < 0.7:
            # Small step: -5 to 5
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
        else:
            # Larger jump: -20 to 20
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            continue
            
        new_point = [new_x, new_y]
        
        # Check distinctness
        is_distinct = True
        for k in range(n):
            if k != idx and points[k][0] == new_x and points[k][1] == new_y:
                is_distinct = False
                break
                
        if not is_distinct:
            continue
            
        # Apply move
        points[idx] = new_point
        new_score = _score(points)
        
        if new_score > current_score:
            # Accept better move
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            # Accept worse move with probability exp(-delta/T)
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                # If score is invalid (-1 or 0 when it was >0), reject strongly
                # But SA logic handles it if T is high enough initially
                if new_score < 0:
                    accept = False
                else:
                    accept = random.random() < math.exp(-delta / T)
                    
            if accept:
                # Keep the move
                pass
            else:
                # Revert
                points[idx] = [old_x, old_y]
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Reheat if stuck for a long time? 
            # For now, just stay at T_min. The random moves will still explore.

    return best_points

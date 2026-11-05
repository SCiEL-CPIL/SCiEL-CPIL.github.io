import random
import time
from math import gcd

def count_ordinary(points):
    """
    Count the number of lines containing exactly two points.
    points: list of (x, y)
    """
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line equation: A*x + B*y + C = 0
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue # Should not happen for distinct points
            
            A //= g
            B //= g
            C //= g
            
            # Canonical form
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_points = None
    best_count = float('inf')
    
    # Initial configurations
    # 1. 3x4 Grid
    grid = [[x, y] for x in range(4) for y in range(3)]
    
    # 2. Random
    def random_points():
        return [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
    
    # 3. Hesse-like (9 pts) + 3 random
    # Hesse points in projective plane, but we need affine.
    # Standard Hesse: (0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), ... 
    # Actually, let's just use random and grid.
    
    initial_configs = [
        grid,
        random_points(),
        random_points()
    ]
    
    for init in initial_configs:
        if time.time() > deadline:
            break
            
        current_points = [p[:] for p in init]
        current_count = count_ordinary(current_points)
        
        if current_count < best_count:
            best_count = current_count
            best_points = [p[:] for p in current_points]
            
        # Local Search
        no_improvement_count = 0
        step_size = 1
        
        while time.time() < deadline and no_improvement_count < 500:
            improved = False
            random.shuffle(current_points) # Randomize order of moves
            
            for i in range(12):
                if time.time() > deadline:
                    break
                    
                x, y = current_points[i]
                
                # Try moving to neighbors
                # Box size increases if stuck?
                # For now, small box.
                moves = []
                for dx in range(-step_size, step_size + 1):
                    for dy in range(-step_size, step_size + 1):
                        if dx == 0 and dy == 0:
                            continue
                        moves.append((x + dx, y + dy))
                        
                # Add some random jumps
                for _ in range(5):
                    moves.append((x + random.randint(-10, 10), y + random.randint(-10, 10)))
                
                for nx, ny in moves:
                    # Check bounds? Not strictly necessary for validity, 
                    # but keeps numbers small.
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                        
                    old_x, old_y = current_points[i]
                    current_points[i] = [nx, ny]
                    
                    # Check distinctness
                    valid = True
                    for j in range(12):
                        if i != j and current_points[i] == current_points[j]:
                            valid = False
                            break
                            
                    if not valid:
                        current_points[i] = [old_x, old_y]
                        continue
                        
                    new_count = count_ordinary(current_points)
                    
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                        no_improvement_count = 0
                        if current_count < best_count:
                            best_count = current_count
                            best_points = [p[:] for p in current_points]
                        # Break inner loop to restart? Or continue?
                        # Continue to see if more improvements possible in this step
                    else:
                        current_points[i] = [old_x, old_y]
                        
            if not improved:
                no_improvement_count += 1
                # Increase step size occasionally
                if no_improvement_count % 10 == 0:
                    step_size += 1
                    if step_size > 5:
                        step_size = 5
                        
    # Return best points found
    if best_points is None:
        return grid
        
    return best_points

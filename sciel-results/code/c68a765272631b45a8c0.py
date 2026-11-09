import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_score(pts):
        # Calculate number of lines with exactly 4 points
        # and check for lines with >= 5 points
        n = len(pts)
        lines = {}
        
        # Map line (normalized) to list of point indices
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                
                # Normalize line representation
                # Use gcd to normalize direction
                if dx == 0 and dy == 0:
                    continue # Points are distinct, so this shouldn't happen if we ensure distinctness
                
                g = abs(__import__('math').gcd(dx, dy))
                if g == 0:
                    g = 1
                ndx = dx // g
                ndy = dy // g
                
                # Ensure consistent sign
                if ndx < 0 or (ndx == 0 and ndy < 0):
                    ndx = -ndx
                    ndy = -ndy
                
                # Line is determined by direction and a point
                # To make it unique, we can use the equation Ax + By = C
                # A = ndy, B = -ndx
                # C = A*x1 + B*y1
                # Normalize A, B, C by gcd(A, B, C)
                A = ndy
                B = -ndx
                C = A * x1 + B * y1
                
                g2 = abs(__import__('math').gcd(__import__('math').gcd(A, B), C))
                if g2 == 0:
                    g2 = 1
                nA = A // g2
                nB = B // g2
                nC = C // g2
                
                # Ensure consistent sign for the line equation
                if nA < 0 or (nA == 0 and nB < 0) or (nA == 0 and nB == 0 and nC < 0):
                    nA = -nA
                    nB = -nB
                    nC = -nC
                    
                line_key = (nA, nB, nC)
                
                if line_key not in lines:
                    lines[line_key] = set()
                lines[line_key].add(i)
                lines[line_key].add(j)
        
        exact_4 = 0
        bad_lines = 0
        for key, idx_set in lines.items():
            if len(idx_set) == 4:
                exact_4 += 1
            elif len(idx_set) >= 5:
                bad_lines += 1
        
        return exact_4 - 1000 * bad_lines

    best_score = get_score(points)
    best_points = [p[:] for p in points]
    
    current_points = [p[:] for p in points]
    current_score = best_score
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        i = random.randint(0, 15)
        old_x, old_y = current_points[i]
        
        # New position
        # Bias towards small integers initially? Or uniform?
        # Uniform in [-1000, 1000]
        new_x = random.randint(-1000, 1000)
        new_y = random.randint(-1000, 1000)
        
        # Ensure distinctness? 
        # If new point equals another, score will be bad (duplicate points might cause issues in line calc? 
        # Actually, if points are not distinct, the problem is invalid. 
        # We should check distinctness.)
        is_distinct = True
        for j in range(16):
            if i != j and current_points[j] == [new_x, new_y]:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        current_points[i] = [new_x, new_y]
        new_score = get_score(current_points)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < __import__('math').exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[i] = [old_x, old_y]
            
        T *= cooling
        if T < 0.1:
            # Restart with random grid or current best?
            # Let's restart from best to explore new basins?
            # Or just continue with low T?
            # Let's do a random restart from the seed grid occasionally
            if random.random() < 0.01:
                current_points = [[x, y] for x in range(4) for y in range(4)]
                current_score = get_score(current_points)
                T = 100.0

    return best_points

# EVOLVE-BLOCK-END

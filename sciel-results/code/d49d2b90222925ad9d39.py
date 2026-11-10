import time
import random
from itertools import combinations

def count_4_lines(points):
    """
    Counts the number of lines containing exactly 4 points.
    points: list of (x, y) tuples.
    """
    if len(points) < 4:
        return 0
    
    # Map from line signature to count of points on that line
    # Line signature: (dx/g, dy/g, c/g) where line is ax+by=c
    # To avoid floating point issues, use integers.
    # Line through (x1, y1) and (x2, y2):
    # dx = x2 - x1, dy = y2 - y1
    # Normalize (dx, dy) by gcd
    # c = y1 * dx - x1 * dy (for equation dx*x - dy*y = c)
    # Wait, standard form: A x + B y + C = 0
    # A = y2 - y1
    # B = x1 - x2
    # C = x2 * y1 - x1 * y2
    # Normalize (A, B, C) by gcd(|A|, |B|, |C|)
    
    lines = {}
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            g = __import__('math').gcd(abs(A), abs(B))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                # Should not happen for distinct points
                continue
                
            # Ensure a consistent sign, e.g., A > 0 or (A == 0 and B > 0)
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            if key in lines:
                lines[key] += 1
            else:
                lines[key] = 1
                
    count = 0
    for key, val in lines.items():
        if val == 4:
            count += 1
            
    return count

def has_five_collinear(points):
    """
    Checks if any 5 points are collinear.
    """
    lines = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = __import__('math').gcd(abs(A), abs(B))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                continue
                
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            if key in lines:
                lines[key] += 1
                if lines[key] == 5:
                    return True
            else:
                lines[key] = 1
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial configurations
    configs = []
    
    # 1. 4x4 Grid
    grid_4x4 = [(x, y) for x in range(4) for y in range(4)]
    configs.append(grid_4x4)
    
    # 2. Random 16 points in 10x10
    random.seed(42)
    for _ in range(5):
        pts = set()
        while len(pts) < 16:
            pts.add((random.randint(0, 9), random.randint(0, 9)))
        configs.append(list(pts))
        
    # 3. 5x5 Grid Subsets (remove 9 points)
    # Try removing corners or random
    grid_5x5 = [(x, y) for x in range(5) for y in range(5)]
    # Remove 9 random points
    for _ in range(5):
        removed = set(random.sample(range(25), 9))
        pts = [grid_5x5[i] for i in range(25) if i not in removed]
        configs.append(pts)
        
    best_score = 0
    best_points = None
    
    for config in configs:
        if has_five_collinear(config):
            continue
        score = count_4_lines(config)
        if score > best_score:
            best_score = score
            best_points = config
            
    # Local Search
    current_points = best_points
    current_score = best_score
    
    if current_points is None:
        # Fallback
        return [[x, y] for x in range(4) for y in range(4)]
        
    # Convert to list of lists for return format later, but work with tuples
    current_points = [tuple(p) for p in current_points]
    
    while time.time() < deadline:
        # Pick a random point
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try moving it
        # Random step in range [-2, 2]
        step_x = random.randint(-3, 3)
        step_y = random.randint(-3, 3)
        
        new_x = old_x + step_x
        new_y = old_y + step_y
        
        # Bounds check
        if new_x < -100 or new_x > 1000 or new_y < -100 or new_y > 1000:
            continue
            
        # Check distinctness
        new_point = (new_x, new_y)
        if any(new_point == p for i, p in enumerate(current_points) if i != idx):
            continue
            
        # Create new set
        trial_points = current_points.copy()
        trial_points[idx] = new_point
        
        # Check 5 collinear
        if has_five_collinear(trial_points):
            continue
            
        # Calculate score
        trial_score = count_4_lines(trial_points)
        
        if trial_score > current_score:
            current_score = trial_score
            current_points = trial_points
            best_score = current_score
            best_points = current_points
        elif trial_score == current_score and random.random() < 0.1:
            # Plateau move
            current_points = trial_points
            
    # Return best found
    return [[x, y] for x, y in best_points]

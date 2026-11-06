import random
import time

def _count_ordinary_lines(points):
    n = len(points)
    lines = {}
    ordinary = 0
    # Map line to count of points
    # Line representation: (A, B, C) normalized
    # To avoid floating point, use integer cross product check or normalized (a,b,c) for ax+by+c=0
    
    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        # Normalize
        if a == 0 and b == 0:
            return None
        g = __import__('math').gcd(a, __import__('math').gcd(b, c))
        if g < 0:
            g = -g
        if g != 0:
            a //= g
            b //= g
            c //= g
        # Ensure consistent sign
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a = -a
            b = -b
            c = -c
        return (a, b, c)

    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = line_key(points[i], points[j])
            if key:
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
                
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial configuration: Triangle with 3 extras per side
    # Vertices: (0,0), (10,0), (0,10)
    # Side 1 (y=0): (1,0), (2,0), (3,0)
    # Side 2 (x=0): (0,1), (0,2), (0,3)
    # Side 3 (x+y=10): (4,6), (5,5), (6,4)
    
    points = [
        [0, 0], [10, 0], [0, 10],
        [1, 0], [2, 0], [3, 0],
        [0, 1], [0, 2], [0, 3],
        [4, 6], [5, 5], [6, 4]
    ]
    
    current_score = _count_ordinary_lines(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local Search
    import random
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random point and move it
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        # Random new position within bounds
        new_x = random.randint(-50, 50)
        new_y = random.randint(-50, 50)
        
        # Ensure distinct
        if any(points[i][0] == new_x and points[i][1] == new_y for i in range(12) if i != idx):
            continue
            
        points[idx] = [new_x, new_y]
        
        new_score = _count_ordinary_lines(points)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        else:
            # Revert
            points[idx] = [old_x, old_y]
            
    return best_points

# EVOLVE-BLOCK-START
def construct():
    # The actual construct function called by evaluator
    # We wrap the logic above
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configuration: Triangle with 3 extras per side
    points = [
        [0, 0], [10, 0], [0, 10],
        [1, 0], [2, 0], [3, 0],
        [0, 1], [0, 2], [0, 3],
        [4, 6], [5, 5], [6, 4]
    ]
    
    current_score = _count_ordinary_lines(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    import random
    random.seed(42)
    
    # Try to improve
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        new_x = random.randint(-50, 50)
        new_y = random.randint(-50, 50)
        
        if any(points[i][0] == new_x and points[i][1] == new_y for i in range(12) if i != idx):
            continue
            
        points[idx] = [new_x, new_y]
        
        new_score = _count_ordinary_lines(points)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        else:
            points[idx] = [old_x, old_y]
            
    return best_points
# EVOLVE-BLOCK-END

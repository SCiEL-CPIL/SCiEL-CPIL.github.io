import time
import random
from fractions import Fraction
from collections import defaultdict

def _count_ordinary_lines(points):
    """
    Counts the number of lines that contain exactly 2 points from the given list.
    Points are tuples of (x, y).
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map each line to the set of points on it
    # A line can be represented by (A, B, C) for Ax + By + C = 0
    # We normalize (A, B, C) such that gcd(A, B, C) = 1 and A >= 0 (or if A==0, B>0)
    
    lines = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            if A == 0 and B == 0:
                # Same point, shouldn't happen if distinct
                continue
                
            # Find GCD
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure canonical form
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            lines[(A, B, C)].add(i)
            
    ordinary_count = 0
    for line_id, point_indices in lines.items():
        if len(point_indices) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) != 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initial Configuration: 3x3 Grid + 3 extra points
    # 3x3 Grid has 12 ordinary lines.
    # Let's try to improve.
    
    best_points = None
    best_count = float('inf')
    
    # Strategy 1: Local Search from 3x4 Grid
    # Strategy 2: Local Search from 3x3 + 3 random
    
    def run_local_search(initial_points, max_steps=10000):
        points = [tuple(p) for p in initial_points]
        current_count = _count_ordinary_lines(points)
        if not _not_all_collinear(points):
            return points, current_count
            
        best_local_points = points
        best_local_count = current_count
        
        no_improve_count = 0
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Pick a random point to move
            idx = random.randint(0, len(points) - 1)
            old_x, old_y = points[idx]
            
            # Generate candidate new positions
            # 1. Random nearby
            # 2. Points that are intersections of existing lines (hard to compute fast)
            # Let's just try random integers in a reasonable range
            # Range [-10, 10] is good for small grids
            
            new_x = random.randint(-10, 10)
            new_y = random.randint(-10, 10)
            
            # Check if new point is distinct from others
            if any((new_x, new_y) == points[i] for i in range(len(points)) if i != idx):
                continue
                
            points[idx] = (new_x, new_y)
            
            # Check collinearity
            if not _not_all_collinear(points):
                points[idx] = (old_x, old_y)
                continue
                
            new_count = _count_ordinary_lines(points)
            
            if new_count < best_local_count:
                best_local_count = new_count
                best_local_points = points[:]
                no_improve_count = 0
            else:
                # Accept with some probability if equal? No, just reject to keep search tight
                # Or accept if equal to allow plateau crossing?
                # For now, reject if not better.
                points[idx] = (old_x, old_y)
                no_improve_count += 1
                
            if no_improve_count > 500:
                break
                
        return best_local_points, best_local_count

    # Initial configurations to try
    inits = []
    
    # 1. 3x4 Grid
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    inits.append(grid_3x4)
    
    # 2. 3x3 Grid + 3 points
    grid_3x3 = [[x, y] for x in range(3) for y in range(3)]
    # Add (0,3), (1,3), (2,3)
    inits.append(grid_3x3 + [[0, 3], [1, 3], [2, 3]])
    
    # 3. 4x3 Grid (transposed)
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    inits.append(grid_4x3)
    
    # 4. Random points
    for _ in range(5):
        random_points = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
        # Ensure distinct
        while len(set(tuple(p) for p in random_points)) < 12:
            random_points[random.randint(0, 11)] = [random.randint(-5, 5), random.randint(-5, 5)]
        inits.append(random_points)
        
    for init in inits:
        if time.time() > deadline:
            break
            
        points, count = run_local_search(init)
        if count < best_count:
            best_count = count
            best_points = points
            
        if best_count <= 6:
            break # Found theoretical minimum
            
    if best_points is None:
        # Fallback to seed
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points

# EVOLVE-BLOCK-END

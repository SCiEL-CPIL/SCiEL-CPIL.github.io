import time
import random
import math
from itertools import combinations
from math import gcd

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two points.
    points: list of (x, y)
    """
    n = len(points)
    line_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C for Ax + By + C = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            
            A //= g
            B //= g
            C //= g
            
            # Standardize sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    ordinary_count = sum(1 for pts in line_points.values() if len(pts) == 2)
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Strategy: Use a Böröczky-like configuration based on regular hexagon + dual
    # The theoretical minimum for 12 points is 6 ordinary lines.
    
    # Approach 1: Regular hexagon vertices + 6 more points arranged symmetrically
    # Scale by an integer to ensure all coordinates are integers
    
    best_points = None
    best_score = float('inf')
    
    # Try multiple scales and offsets to find integer configurations
    # Start with the classic Böröczky construction idea
    
    # A regular hexagon centered at origin with "radius" r has vertices:
    # (r, 0), (r/2, r*sqrt(3)/2), (-r/2, r*sqrt(3)/2), (-r, 0), (-r/2, -r*sqrt(3)/2), (r/2, -r*sqrt(3)/2)
    
    # For integer coordinates, we can use a scaled version.
    # Let's try different constructions:
    
    # Construction 1: Two concentric hexagons (inner and outer)
    # Outer hexagon: scale by 2 to get integers
    # Inner hexagon: smaller scale
    
    # Actually, let's try a more systematic approach:
    # Use points that lie on a few lines to minimize ordinary lines.
    
    # Known good configuration: 4 points on each of 3 concurrent lines, with no 4 collinear
    # This gives C(4,2)*3 = 18 lines total, but many are ordinary.
    
    # Better: Use the fact that if we have k points on a line, that line is not ordinary.
    # We want to maximize the number of non-ordinary lines.
    
    # Let's try: 3 lines, each with 4 points, all concurrent at one point.
    # But we need 12 distinct points, and the center is shared.
    # So: 3 lines through origin, each with 5 points (including origin), but origin counted once.
    # Total: 1 + 3*4 = 13 points. Too many.
    
    # Try: 2 lines with 6 points each, intersecting at one point.
    # Total: 6 + 6 - 1 = 11 points. Need 1 more.
    
    # Let me just try random search with the hexagon as initial guess.
    
    # Hexagon with integer coordinates (approximate):
    # Using a 30-60-90 triangle scaled: (2, 0), (1, 3), (-1, 3), (-2, 0), (-1, -3), (1, -3)
    # This is not regular but close.
    
    hexagon = [(2, 0), (1, 3), (-1, 3), (-2, 0), (-1, -3), (1, -3)]
    
    # Add 6 more points to form a good configuration
    # Try points at (0, 0) and symmetric positions
    
    # Let's try a known Böröczky configuration for n=12
    # The Böröczky example uses points on two conics.
    
    # For simplicity, let's do a local search starting from a reasonable initial config.
    
    # Initial: 3x4 grid
    initial_points = [(x, y) for x in range(4) for y in range(3)]
    
    # Also try the hexagon-based config
    hex_based = hexagon + [(0, 0), (0, 6), (0, -6), (3, 0), (-3, 0), (0, 12)]
    
    candidates = [initial_points, hex_based]
    
    current_points = [list(p) for p in candidates[0]]
    current_score = count_ordinary_lines(current_points)
    
    best_points = [list(p) for p in current_points]
    best_score = current_score
    
    # Local search
    def try_move(points, i, new_x, new_y):
        old_x, old_y = points[i]
        if old_x == new_x and old_y == new_y:
            return points, False
        
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            return points, False
            
        for j in range(len(points)):
            if j != i:
                if points[j][0] == new_x and points[j][1] == new_y:
                    return points, False
        
        points[i][0] = new_x
        points[i][1] = new_y
        return points, True
    
    # Try each candidate as starting point
    for cand_idx, cand in enumerate(candidates):
        if time.time() >= deadline:
            break
            
        current_points = [list(p) for p in cand]
        current_score = count_ordinary_lines(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [list(p) for p in current_points]
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() >= deadline:
                    break
                cx, cy = current_points[i]
                
                # Try random moves
                for _ in range(30):
                    if time.time() >= deadline:
                        break
                    nx = cx + random.randint(-10, 10)
                    ny = cy + random.randint(-10, 10)
                    
                    new_points, valid = try_move(current_points, i, nx, ny)
                    if not valid:
                        continue
                        
                    new_score = count_ordinary_lines(new_points)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [list(p) for p in current_points]
                        break
                    else:
                        # Revert
                        current_points[i][0] = cx
                        current_points[i][1] = cy
    
    # Final verification
    result = [[int(p[0]), int(p[1])] for p in best_points]
    
    # Ensure distinctness and bounds
    seen = set()
    new_result = []
    for p in result:
        tp = (p[0], p[1])
        if tp not in seen:
            seen.add(tp)
            new_result.append(p)
    result = new_result
    
    return result

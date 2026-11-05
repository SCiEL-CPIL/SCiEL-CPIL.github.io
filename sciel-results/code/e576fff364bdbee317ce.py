import time
import math
from math import gcd
from itertools import product

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_point_indices = {}
    
    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        g = gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
        c = dy * x1 - dx * y1
        return (dx, dy, c)
    
    for i in range(n):
        for j in range(i+1, n):
            key = line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_point_indices:
                line_point_indices[key] = set()
            line_point_indices[key].add(i)
            line_point_indices[key].add(j)
    
    ordinary = 0
    for key, idx_set in line_point_indices.items():
        if len(idx_set) == 2:
            ordinary += 1
            
    return ordinary

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(points, count):
        nonlocal best_points, best_count
        if count < best_count:
            best_count = count
            best_points = [p[:] for p in points]

    # Strategy: Parametric Algebraic Construction
    # We suspect the optimal configuration is a deformation of the 3x4 grid.
    # The 3x4 grid has points (x,y) with x in {0,1,2}, y in {0,1,2,3}.
    # Let's generalize the x-coordinates for each row to be variable.
    # Row 0: (a0, 0), (a1, 0), (a2, 0)
    # Row 1: (b0, 1), (b1, 1), (b2, 1)
    # Row 2: (c0, 2), (c1, 2), (c2, 2)
    # Row 3: (d0, 3), (d1, 3), (d2, 3)
    
    # To maintain "few ordinary lines", we want many diagonals to have 3+ points.
    # A key insight from Böröczky configurations: 
    # If we take a 3x4 grid, it has 6 ordinary lines? Let's verify.
    # Actually, let's just search a small neighborhood of the 3x4 grid in integer coordinates.
    # But the space is huge.
    
    # Alternative: Use the "Pappus" or "Desargues" like structures?
    # For n=12, the minimum is 6.
    # A known configuration achieving 6 ordinary lines for n=12 is derived from the affine plane AG(2,3) 
    # plus 3 more points? No, AG(2,3) has 9 points.
    
    # Let's try a specific construction:
    # 4 lines of 3 points.
    # L1: y=0, x in {0, 1, 2}
    # L2: y=1, x in {0, 1, 2}
    # L3: y=2, x in {0, 1, 2}
    # L4: y=3, x in {0, 1, 2}
    # This is the 3x4 grid.
    
    # Let's check the count for the 3x4 grid first.
    grid_34 = [[x, y] for x in range(3) for y in range(4)]
    cnt = count_ordinary_lines(grid_34)
    update_best(grid_34, cnt)
    
    # If the 3x4 grid has more than 6, we need to deform it.
    # Deformation idea:
    # Keep y coordinates as 0, 1, 2, 3.
    # Vary x coordinates.
    # Let row y have x-coordinates x_{y,0}, x_{y,1}, x_{y,2}.
    # To create a diagonal with 3 points, e.g., (x_{0,0}, 0), (x_{1,1}, 1), (x_{2,2}, 2) must be collinear.
    # This implies x_{1,1} - x_{0,0} = x_{2,2} - x_{1,1} => 2*x_{1,1} = x_{0,0} + x_{2,2}.
    
    # Let's do a targeted search over small integer shifts.
    # Base: 3x4 grid.
    # We will try shifting individual points slightly.
    # But random local search failed to improve in the seed.
    
    # Let's try a different structural family:
    # "2x6" grid?
    grid_26 = [[x, y] for x in range(2) for y in range(6)]
    cnt = count_ordinary_lines(grid_26)
    update_best(grid_26, cnt)
    
    # "4x3" grid?
    grid_43 = [[x, y] for x in range(4) for y in range(3)]
    cnt = count_ordinary_lines(grid_43)
    update_best(grid_43, cnt)
    
    # "6x2" grid?
    grid_62 = [[x, y] for x in range(6) for y in range(2)]
    cnt = count_ordinary_lines(grid_62)
    update_best(grid_62, cnt)
    
    # Let's try a configuration based on a regular polygon approximation?
    # No, integer coords.
    
    # Let's try a "star" configuration?
    # Center (0,0).
    # 6 rays. 2 points per ray?
    # This creates many ordinary lines from center to points?
    # No, if 2 points on a ray with center, that's 3 collinear.
    # Points: (0,0), (1,0), (2,0), (0,1), (0,2), (-1,0), (-2,0), (0,-1), (0,-2), (1,1), (2,2), (-1,-1), (-2,-2) -> 13 points.
    # We need 12.
    # Remove center?
    # 6 lines of 2 points passing through origin?
    # If we have 12 points on 6 lines through origin, each line has 2 points (one on each side? or both on same side?).
    # If 2 points on same ray, distance 1 and 2.
    # Line through (1,0) and (2,0) is x-axis. (0,0) is not in set.
    # So x-axis has 2 points. Ordinary.
    # This would yield 6 ordinary lines if no other collinearities.
    # But we need to check cross-collinearities.
    
    # Configuration:
    # 6 lines through origin, each with 2 points at distance 1 and 2?
    # No, if points are (1,0) and (2,0), the line is the x-axis.
    # If we have 6 such pairs, we have 12 points.
    # Are there other collinearities?
    # Points: (1,0), (2,0), (0,1), (0,2), (-1,0), (-2,0), (0,-1), (0,-2), (1,1), (2,2), (-1,-1), (-2,-2).
    # Check line (1,0) and (0,1): x+y=1. Any other points?
    # (2,0) -> 2+0=2. No.
    # (0,2) -> 0+2=2. No.
    # (-1,0) -> -1. No.
    # (1,1) -> 2. No.
    # So (1,0)-(0,1) is an ordinary line.
    # How many such "cross" lines?
    # This configuration likely has many ordinary lines.
    
    # Let's try to improve the 3x4 grid using a specific algebraic perturbation.
    # The 3x4 grid has points (0,0)..(2,3).
    # Ordinary lines in 3x4 grid:
    # Slopes 0, inf, 1, -1, 1/2, -1/2, 2, -2, 3, -3, 1/3, -1/3...
    # Let's just run a very fast local search with a better move generator.
    # Move: Swap two points?
    # Move: Shift a whole row?
    
    def local_improve_structured(points, max_iters=5000):
        pts = [p[:] for p in points]
        current_count = count_ordinary_lines(pts)
        
        # Identify rows and columns if grid-like
        # For simplicity, just random point moves with larger steps
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            i = int(random.random() * 12)
            old_x, old_y = pts[i]
            # Try moves in a 5x5 neighborhood
            found_move = False
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check distinctness
                    pts[i] = [new_x, new_y]
                    if len(set(tuple(p) for p in pts)) < 12:
                        continue
                    new_count = count_ordinary_lines(pts)
                    if new_count < current_count:
                        current_count = new_count
                        found_move = True
                        break
                    else:
                        pts[i] = [old_x, old_y]
                if found_move:
                    break
            if not found_move:
                # If no improvement in 5x5, try a larger random jump
                new_x = old_x + int(random.gauss(0, 10))
                new_y = old_y + int(random.gauss(0, 10))
                pts[i] = [new_x, new_y]
                if len(set(tuple(p) for p in pts)) == 12:
                    new_count = count_ordinary_lines(pts)
                    if new_count < current_count:
                        current_count = new_count
                    else:
                        pts[i] = [old_x, old_y]
        
        return pts, current_count

    # Run structured local search on the best grid found so far
    if best_points:
        improved_pts, improved_cnt = local_improve_structured(best_points, max_iters=10000)
        update_best(improved_pts, improved_cnt)
        
    # If we still haven't beaten the theoretical minimum (6), we return the best found.
    # Note: The 3x4 grid is known to have 6 ordinary lines?
    # Let's assume the local search will find it or confirm it.
    
    if best_points is None:
        return grid_34
        
    return best_points

import random

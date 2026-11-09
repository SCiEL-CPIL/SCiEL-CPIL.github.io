import time
import sys
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Use a set of normalized line representations
    lines = set()
    ordinary = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Normalize the line: (dx, dy) where gcd(dx, dy) = 1
            dx = x2 - x1
            dy = y2 - y1
            
            # Handle vertical lines
            if dx == 0:
                line_key = ('v', x1)
            elif dy == 0:
                line_key = ('h', y1)
            else:
                # Normalize direction
                from math import gcd
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                
                # Ensure consistent sign
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                elif dx_n == 0 and dy_n < 0:
                    dy_n = -dy_n
                
                # Line equation: dy*x - dx*y = c
                c = dy_n * x1 - dx_n * y1
                line_key = (dx_n, dy_n, c)
            
            lines.add(line_key)
    
    # Count how many points are on each line
    line_point_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0:
                line_key = ('v', x1)
            elif dy == 0:
                line_key = ('h', y1)
            else:
                from math import gcd
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                elif dx_n == 0 and dy_n < 0:
                    dy_n = -dy_n
                
                c = dy_n * x1 - dx_n * y1
                line_key = (dx_n, dy_n, c)
            
            if line_key not in line_point_counts:
                line_point_counts[line_key] = set()
            line_point_counts[line_key].add(i)
            line_point_counts[line_key].add(j)
    
    # Count ordinary lines (exactly 2 points)
    ordinary = sum(1 for pts in line_point_counts.values() if len(pts) == 2)
    
    return ordinary

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Uses systematic search over structured configurations.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_ordinary = float('inf')
    
    # Try various structured configurations
    # Configuration 1: Grid-like patterns
    
    # 3x4 grid
    points_34 = [[i, j] for i in range(3) for j in range(4)]
    ord_count = _count_ordinary_lines(points_34)
    if ord_count < best_ordinary:
        best_ordinary = ord_count
        best_points = points_34
    
    # 4x3 grid
    points_43 = [[i, j] for i in range(4) for j in range(3)]
    ord_count = _count_ordinary_lines(points_43)
    if ord_count < best_ordinary:
        best_ordinary = ord_count
        best_points = points_43
    
    # 2x6 grid
    points_26 = [[i, j] for i in range(2) for j in range(6)]
    ord_count = _count_ordinary_lines(points_26)
    if ord_count < best_ordinary:
        best_ordinary = ord_count
        best_points = points_26
    
    # 6x2 grid
    points_62 = [[i, j] for i in range(6) for j in range(2)]
    ord_count = _count_ordinary_lines(points_62)
    if ord_count < best_ordinary:
        best_ordinary = ord_count
        best_points = points_62
    
    # Check time
    if time.time() > deadline:
        return best_points if best_points else [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Try configurations with points on fewer lines
    # 6 points on x-axis, 6 points on another line
    
    # Two parallel lines with 6 points each
    for offset in [1, 2, 3]:
        points_two = [[i, 0] for i in range(6)] + [[i, offset] for i in range(6)]
        ord_count = _count_ordinary_lines(points_two)
        if ord_count < best_ordinary:
            best_ordinary = ord_count
            best_points = points_two
    
    # Check time
    if time.time() > deadline:
        return best_points if best_points else [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Try 4 lines with 3 points each
    # Horizontal lines
    for y1, y2, y3, y4 in [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4)]:
        points_h = []
        for y in [y1, y2, y3, y4]:
            points_h.extend([[x, y] for x in range(3)])
        
        # Check for duplicates
        if len(set(tuple(p) for p in points_h)) == 12:
            ord_count = _count_ordinary_lines(points_h)
            if ord_count < best_ordinary:
                best_ordinary = ord_count
                best_points = points_h
    
    # Check time
    if time.time() > deadline:
        return best_points if best_points else [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Try configurations mixing different line directions
    # 3 points on x-axis, 3 on y-axis, 3 on x=1, 3 on y=1
    points_mixed = []
    for x in range(3):
        points_mixed.append([x, 0])
    for y in range(1, 4):
        points_mixed.append([0, y])
    for y in range(1, 4):
        points_mixed.append([1, y])
    for x in range(1, 4):
        points_mixed.append([x, 1])
    
    # Remove duplicates
    seen = set()
    unique_points = []
    for p in points_mixed:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_points.append(p)
    
    if len(unique_points) == 12:
        ord_count = _count_ordinary_lines(unique_points)
        if ord_count < best_ordinary:
            best_ordinary = ord_count
            best_points = unique_points
    
    # Check time
    if time.time() > deadline:
        return best_points if best_points else [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Try more structured configurations
    # Points forming a pattern with more collinearities
    
    # 2 points on each of 6 different lines through origin
    import math
    angles = [0, 30, 60, 90, 120, 150]
    points_radial = []
    for angle_deg in angles:
        angle = math.radians(angle_deg)
        for dist in [1, 2]:
            x = round(dist * math.cos(angle))
            y = round(dist * math.sin(angle))
            if (x, y) not in [(p[0], p[1]) for p in points_radial]:
                points_radial.append([x, y])
    
    if len(points_radial) == 12:
        ord_count = _count_ordinary_lines(points_radial)
        if ord_count < best_ordinary:
            best_ordinary = ord_count
            best_points = points_radial
    
    # Check time
    if time.time() > deadline:
        return best_points if best_points else [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Try affine transformations of known good configurations
    # Start with 3x4 grid and try different spacings
    
    for sx in [1, 2, 3]:
        for sy in [1, 2, 3]:
            points_scaled = [[i * sx, j * sy] for i in range(3) for j in range(4)]
            ord_count = _count_ordinary_lines(points_scaled)
            if ord_count < best_ordinary:
                best_ordinary = ord_count
                best_points = points_scaled
        
        if time.time() > deadline:
            return best_points if best_points else [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Try sheared grids
    for shear in [1, 2]:
        points_sheared = [[i + j * shear, j] for i in range(3) for j in range(4)]
        ord_count = _count_ordinary_lines(points_sheared)
        if ord_count < best_ordinary:
            best_ordinary = ord_count
            best_points = points_sheared
    
    # Check time
    if time.time() > deadline:
        return best_points if best_points else [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Try configurations based on finite field geometry
    # 12 points in a structured pattern
    
    # 4 rows of 3 points with increasing offset
    for offset_pattern in [(0, 0, 0, 0), (0, 1, 2, 3), (0, 1, 1, 2)]:
        points_offset = []
        for row in range(4):
            for col in range(3):
                x = col + offset_pattern[row]
                y = row
                points_offset.append([x, y])
        
        if len(set(tuple(p) for p in points_offset)) == 12:
            ord_count = _count_ordinary_lines(points_offset)
            if ord_count < best_ordinary:
                best_ordinary = ord_count
                best_points = points_offset
    
    # Check time
    if time.time() > deadline:
        return best_points if best_points else [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Final fallback
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
    
    return best_points

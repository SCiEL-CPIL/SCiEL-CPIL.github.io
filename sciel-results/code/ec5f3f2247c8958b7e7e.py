import random
import math

def count_collinear_fast(points):
    """
    Returns (c3, c4) where c3 is number of 3-collinear triples and c4 is number of 4-collinear quadruples.
    Uses O(N^2) slope counting.
    """
    n = len(points)
    c3_total = 0
    c4_total = 0
    
    for i in range(n):
        slopes = {}
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize slope (dx, dy)
            # Handle vertical lines
            if dx == 0:
                key = (0, 1)
            elif dy == 0:
                key = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                # Ensure consistent sign
                if dx < 0:
                    dx = -dx
                    dy = -dy
                key = (dx, dy)
            if key not in slopes:
                slopes[key] = 0
            slopes[key] += 1
        
        for count in slopes.values():
            # count is number of other points on this line through i
            # Total points on this line = count + 1
            # Number of 3-subsets containing i on this line = C(count, 2)
            c3_total += count * (count - 1) // 2
            # Number of 4-subsets containing i on this line = C(count, 3)
            if count >= 3:
                c4_total += count * (count - 1) * (count - 2) // 6
                
    # Each 3-collinear triple is counted 3 times (once for each point)
    c3_total //= 3
    # Each 4-collinear quadruple is counted 4 times (once for each point)
    c4_total //= 4
    
    return c3_total, c4_total

def construct():
    # Precomputed configuration: Two 3x3 grids and two isolated points
    # Grid 1: x in [0, 2], y in [0, 2]
    # Grid 2: x in [10, 12], y in [10, 12]
    # Extra 1: (5, 6)
    # Extra 2: (15, 16)
    
    points = []
    # Grid 1
    for x in range(3):
        for y in range(3):
            points.append((x, y))
            
    # Grid 2
    for x in range(10, 13):
        for y in range(10, 13):
            points.append((x, y))
            
    # Extra points
    points.append((5, 6))
    points.append((15, 16))
    
    return [[p[0], p[1]] for p in points]

import random
import math

N = 64

def get_forbidden_points(a, b):
    """
    Returns a set of points (x, y) that form an isosceles triangle with a and b.
    a, b are tuples (x, y).
    """
    x1, y1 = a
    x2, y2 = b
    dx = x2 - x1
    dy = y2 - y1
    d_sq = dx * dx + dy * dy
    
    forbidden = set()
    
    # 1. Perpendicular Bisector
    # 2*dx*x + 2*dy*y = x2^2 - x1^2 + y2^2 - y1^2
    # 2*dx*x + 2*dy*y = C
    C = (x2 * x2 - x1 * x1) + (y2 * y2 - y1 * y1)
    
    if dx == 0 and dy == 0:
        return forbidden # Should not happen as a != b
        
    if dx == 0:
        # dy*y = C/2 => y = C / (2*dy)
        if C % (2 * dy) == 0:
            y = C // (2 * dy)
            if 0 <= y < N:
                for x in range(N):
                    forbidden.add((x, y))
    elif dy == 0:
        # dx*x = C/2 => x = C / (2*dx)
        if C % (2 * dx) == 0:
            x = C // (2 * dx)
            if 0 <= x < N:
                for y in range(N):
                    forbidden.add((x, y))
    else:
        # 2*dx*x + 2*dy*y = C
        # We can iterate x and solve for y
        for x in range(N):
            rem = C - 2 * dx * x
            if rem % (2 * dy) == 0:
                y = rem // (2 * dy)
                if 0 <= y < N:
                    forbidden.add((x, y))

    # 2. Circle centered at a with radius d
    # (x-x1)^2 + (y-y1)^2 = d_sq
    # Iterate x
    # (x-x1)^2 <= d_sq
    # |x-x1| <= sqrt(d_sq)
    max_dx = int(math.isqrt(d_sq))
    
    for x in range(max(0, x1 - max_dx), min(N, x1 + max_dx + 1)):
        dy_sq = d_sq - (x - x1) * (x - x1)
        if dy_sq < 0:
            continue
        dy = int(math.isqrt(dy_sq))
        if dy * dy == dy_sq:
            y1_val = y1 - dy
            y2_val = y1 + dy
            if 0 <= y1_val < N:
                forbidden.add((x, y1_val))
            if 0 <= y2_val < N:
                forbidden.add((x, y2_val))

    # 3. Circle centered at b with radius d
    # (x-x2)^2 + (y-y2)^2 = d_sq
    for x in range(max(0, x2 - max_dx), min(N, x2 + max_dx + 1)):
        dy_sq = d_sq - (x - x2) * (x - x2)
        if dy_sq < 0:
            continue
        dy = int(math.isqrt(dy_sq))
        if dy * dy == dy_sq:
            y1_val = y2 - dy
            y2_val = y2 + dy
            if 0 <= y1_val < N:
                forbidden.add((x, y1_val))
            if 0 <= y2_val < N:
                forbidden.add((x, y2_val))

    return forbidden

def construct():
    best_points = []
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Try multiple random orders
    for _ in range(50):
        random.shuffle(all_points)
        points = []
        forbidden = set()
        
        for p in all_points:
            if p not in forbidden:
                points.append(p)
                # Update forbidden set with new pairs involving p
                # p is the last one added, so pairs are (p, q) for q in points[:-1]
                for q in points[:-1]:
                    forbidden.update(get_forbidden_points(p, q))
        
        if len(points) > len(best_points):
            best_points = points
            
    return best_points

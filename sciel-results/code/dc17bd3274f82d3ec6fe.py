import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Known 5-point set
    base_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Translate p1 to (0,0)
        x2, y2 = p2[0]-x1, p2[1]-y1
        x3, y3 = p3[0]-x1, p3[1]-y1
        x4, y4 = p4[0]-x1, p4[1]-y1
        
        d = (x2 * (y3 * (x4*x4 + y4*y4) - y4 * (x3*x3 + y3*y3)) -
             y2 * (x3 * (x4*x4 + y4*y4) - x4 * (x3*x3 + y3*y3)) +
             (x2*x2 + y2*y2) * (x3 * y4 - x4 * y3))
        return d == 0

    def is_valid_set(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        
        # Check collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        
        # Check concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Strategy: Try to extend the base set
    # Generate candidates in a large box
    # Check if candidate has integer distance to all base points
    # Then check if the new set is valid
    
    best_set = base_set
    best_size = 5
    
    # Candidate box
    # The base points are roughly in [-56, 16] x [0, 63].
    # Let's search a box around this region, e.g., [-100, 100]
    
    for box in [100, 200]:
        if time.time() > deadline - 5.0:
            break
            
        candidates = []
        for x in range(-box, box + 1):
            for y in range(-box, box + 1):
                # Quick check: integer distance to all base points
                valid_dist = True
                for bx, by in base_set:
                    dx = x - bx
                    dy = y - by
                    d2 = dx*dx + dy*dy
                    if not is_square(d2):
                        valid_dist = False
                        break
                if valid_dist:
                    candidates.append((x, y))
        
        # Now check each candidate
        for p in candidates:
            if time.time() > deadline - 2.0:
                break
            new_set = base_set + [p]
            if is_valid_set(new_set):
                best_set = new_set
                best_size = 6
                break
        if best_size == 6:
            break
    
    # If no 6-point set found, try random 6-clique search in a smaller box?
    # Or just return 5.
    
    D = 1
    out_points = []
    for x, y in best_set:
        out_points.append([[x, 1], [y, 1]])
        
    return {'D': D, 'points': out_points}

if __name__ == "__main__":
    res = construct()
    print(res)

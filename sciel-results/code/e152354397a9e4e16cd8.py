import math
import time

def construct():
    start_time = time.time()
    # Initial seed points
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_valid_point(p, pts):
        """
        Checks if point p is valid to add to the set pts.
        1. Distances to all points in pts are integers.
        2. No three points are collinear.
        3. No four points are concyclic.
        """
        x, y = p
        
        # 1. Distances
        for q in pts:
            dx = x - q[0]
            dy = y - q[1]
            d2 = dx*dx + dy*dy
            s = int(math.isqrt(d2))
            if s*s != d2:
                return False
            if d2 == 0:
                return False
        
        # 2. Collinearity
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                p1 = pts[i]
                p2 = pts[j]
                # Cross product
                if (p1[0]-x)*(p2[1]-y) - (p1[1]-y)*(p2[0]-x) == 0:
                    return False
        
        # 3. Concyclicity
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                for k in range(j+1, len(pts)):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    
                    A1 = p1[0]**2 + p1[1]**2
                    A2 = p2[0]**2 + p2[1]**2
                    A3 = p3[0]**2 + p3[1]**2
                    
                    det = (p1[0]-p3[0])*(p2[1]-p3[1]) - (p1[1]-p3[1])*(p2[0]-p3[0])
                    if det == 0:
                        continue
                    
                    D_num = (A3-A1)*(p2[1]-p3[1]) - (A3-A2)*(p1[1]-p3[1])
                    E_num = (p1[0]-p3[0])*(A3-A2) - (p2[0]-p3[0])*(A3-A1)
                    F_num = A1*det - D_num*p1[0] - E_num*p1[1]
                    
                    val = det*(x*x + y*y) + D_num*x + E_num*y + F_num
                    if val == 0:
                        return False
        
        return True

    current_points = base_points
    
    # Search for new points using Pythagorean triples
    # We assume the new point has integer distance to (0,0)
    # This is a heuristic.
    
    # To increase chances, we can also try shifting the coordinate system
    # or just try a grid search if Pythagorean fails.
    
    # Let's try Pythagorean search first
    found_any = False
    for round_num in range(20):
        if time.time() - start_time > 100:
            break
        
        found = False
        # Increase search range
        max_m = 20 + round_num * 5
        max_k = 5 + round_num * 2
        
        for m in range(2, max_m):
            for n in range(1, m):
                if (m-n) % 2 == 0: continue
                if math.gcd(m, n) != 1: continue
                x0 = m*m - n*n
                y0 = 2*m*n
                
                for k in range(1, max_k):
                    x = k * x0
                    y = k * y0
                    candidates = [
                        (x, y), (x, -y), (-x, y), (-x, -y),
                        (y, x), (y, -x), (-y, x), (-y, -x)
                    ]
                    
                    for p in candidates:
                        if is_valid_point(p, current_points):
                            current_points.append(p)
                            found = True
                            found_any = True
                            break
                    if found: break
                if found: break
            if found: break
        
        if not found:
            # Try a grid search in a small area around the centroid
            # This might catch points that don't have integer distance to origin
            if time.time() - start_time > 110:
                break
            
            xs = [p[0] for p in current_points]
            ys = [p[1] for p in current_points]
            cx = sum(xs) // len(xs)
            cy = sum(ys) // len(ys)
            
            grid_radius = 50
            grid_found = False
            for dx in range(-grid_radius, grid_radius):
                if time.time() - start_time > 115: break
                for dy in range(-grid_radius, grid_radius):
                    p = (cx + dx, cy + dy)
                    if is_valid_point(p, current_points):
                        current_points.append(p)
                        grid_found = True
                        break
                if grid_found: break
            
            if not grid_found:
                break

    D = 1
    points_out = []
    for x, y in current_points:
        points_out.append([[x, 1], [y, 1]])
        
    return {'D': D, 'points': points_out}

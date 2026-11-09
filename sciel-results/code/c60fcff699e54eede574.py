def construct():
    import math
    import time
    import random
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 145
    
    # Base set of 6 known valid points
    base_points = [(0, 0), (56, 33), (16, 63), (0, 33), (16, -63), (-16, -63)]
    
    def check_distance(x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        d2 = dx*dx + dy*dy
        r = math.isqrt(d2)
        return r*r == d2
    
    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        area2 = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        return area2 == 0
    
    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        a1 = x2-x1; b1 = y2-y1; c1 = -((x2*x2+y2*y2) - (x1*x1+y1*y1))
        a2 = x3-x1; b2 = y3-y1; c2 = -((x3*x3+y3*y3) - (x1*x1+y1*y1))
        det = a1*b2 - a2*b1
        if det == 0:
            return False
        val = det*(x4*x4+y4*y4 - x1*x1-y1*y1) + (c1*b2-c2*b1)*(x4-x1) + (a1*c2-a2*c1)*(y4-y1)
        return val == 0
    
    def is_valid_set(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if not check_distance(points[i][0], points[i][1], points[j][0], points[j][1]):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True
    
    if not is_valid_set(base_points):
        base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_set = base_points[:]
    n_base = len(best_set)
    
    # Strategy: Use rational parametrization of unit circle to generate candidates
    # Points on a circle centered at origin: (r*(1-t^2)/(1+t^2), r*2t/(1+t^2))
    # For rational t, these have rational coordinates
    # We need distances between all pairs to be rational
    
    # Generate candidate points using rational parametrization
    # For a circle of radius R centered at (cx, cy), parametrize with rational t
    # x = cx + R*(1-t^2)/(1+t^2), y = cy + R*2t/(1+t^2)
    
    # We'll try multiple circles and radii to find good candidates
    
    def generate_candidates_on_circle(cx, cy, R, max_t_denom):
        """Generate rational points on circle using parametrization t = p/q"""
        candidates = []
        for q in range(1, max_t_denom + 1):
            for p in range(-max_t_denom * q, max_t_denom * q + 1):
                if math.gcd(p, q) != 1:
                    continue
                t = Fraction(p, q)
                t2 = t * t
                denom = 1 + t2
                x = cx + R * (1 - t2) / denom
                y = cy + R * 2 * t / denom
                # Convert to integers if possible
                x_frac = x
                y_frac = y
                if x_frac.denominator == 1 and y_frac.denominator == 1:
                    candidates.append((x_frac.numerator, y_frac.numerator))
        return candidates
    
    # Try to find additional points by searching in a bounded region
    # First, try the original approach with numpy for speed
    import numpy as np
    
    for search_range in [100, 200, 400, 700, 1200, 2000]:
        if time.time() > deadline:
            break
        
        xs = np.arange(-search_range, search_range+1, dtype=np.int64)
        ys = np.arange(-search_range, search_range+1, dtype=np.int64)
        X, Y = np.meshgrid(xs, ys)
        candidates = np.column_stack([X.ravel(), Y.ravel()])
        
        mask = np.ones(len(candidates), dtype=bool)
        for px, py in best_set:
            mask &= ~((candidates[:,0] == px) & (candidates[:,1] == py))
        candidates = candidates[mask]
        
        if len(candidates) == 0:
            continue
        
        valid = np.ones(len(candidates), dtype=bool)
        for px, py in best_set:
            dx = candidates[:,0] - px
            dy = candidates[:,1] - py
            d2 = dx*dx + dy*dy
            r_floor = np.sqrt(d2).astype(np.int64)
            valid &= (d2 == r_floor*r_floor) | (d2 == (r_floor+1)*(r_floor+1))
        
        candidates = candidates[valid]
        
        if len(candidates) == 0:
            continue
        
        # Sort candidates by some heuristic (e.g., distance from origin)
        dists = np.sum(candidates**2, axis=1)
        order = np.argsort(dists)
        candidates = candidates[order]
        
        for idx in order:
            if time.time() > deadline:
                break
            x = int(candidates[idx, 0]); y = int(candidates[idx, 1])
            
            valid_collinear = True
            for i in range(n_base):
                for j in range(i+1, n_base):
                    if check_collinear(best_set[i], best_set[j], (x, y)):
                        valid_collinear = False
                        break
                if not valid_collinear:
                    break
            if not valid_collinear:
                continue
            
            valid_concyclic = True
            for i in range(n_base):
                for j in range(i+1, n_base):
                    for k in range(j+1, n_base):
                        if check_concyclic(best_set[i], best_set[j], best_set[k], (x, y)):
                            valid_concyclic = False
                            break
                    if not valid_concyclic:
                        break
                if not valid_concyclic:
                    break
            if not valid_concyclic:
                continue
            
            candidate = best_set[:] + [(x, y)]
            if is_valid_set(candidate):
                best_set = candidate
                n_base = len(best_set)
                break
    
    # Also try using rational parametrization for more targeted search
    if time.time() < deadline:
        # Try circles centered at origin with various radii
        for R in [35, 50, 65, 80, 100]:
            if time.time() > deadline:
                break
            cands = generate_candidates_on_circle(0, 0, R, 50)
            # Filter out points already in best_set
            cands = [p for p in cands if p not in best_set]
            for x, y in cands:
                if time.time() > deadline:
                    break
                valid_collinear = True
                for i in range(n_base):
                    for j in range(i+1, n_base):
                        if check_collinear(best_set[i], best_set[j], (x, y)):
                            valid_collinear = False
                            break
                    if not valid_collinear:
                        break
                if not valid_collinear:
                    continue
                
                valid_concyclic = True
                for i in range(n_base):
                    for j in range(i+1, n_base):
                        for k in range(j+1, n_base):
                            if check_concyclic(best_set[i], best_set[j], best_set[k], (x, y)):
                                valid_concyclic = False
                                break
                        if not valid_concyclic:
                            break
                    if not valid_concyclic:
                        break
                if not valid_concyclic:
                    continue
                
                candidate = best_set[:] + [(x, y)]
                if is_valid_set(candidate):
                    best_set = candidate
                    n_base = len(best_set)
                    break
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

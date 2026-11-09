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
        if d2 == 0:
            return False
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
    
    def is_valid_incremental(existing, new_pt):
        """Check if new_pt can be added to existing set"""
        x, y = new_pt
        n = len(existing)
        
        # Check distances to all existing points
        for px, py in existing:
            if not check_distance(x, y, px, py):
                return False
        
        # Check collinearity with all pairs
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(existing[i], existing[j], new_pt):
                    return False
        
        # Check concyclicity with all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(existing[i], existing[j], existing[k], new_pt):
                        return False
        
        return True
    
    if not is_valid_set(base_points):
        base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_set = base_points[:]
    n_base = len(best_set)
    
    # Generate candidate points using rational parametrization
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
                x_frac = x
                y_frac = y
                if x_frac.denominator == 1 and y_frac.denominator == 1:
                    candidates.append((x_frac.numerator, y_frac.numerator))
        return candidates
    
    # Try to find additional points by searching in a bounded region
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
        
        # Sort candidates by distance from centroid of existing points for better heuristic
        if n_base > 0:
            cx = sum(p[0] for p in best_set) / n_base
            cy = sum(p[1] for p in best_set) / n_base
            dists = (candidates[:,0] - cx)**2 + (candidates[:,1] - cy)**2
        else:
            dists = np.sum(candidates**2, axis=1)
        order = np.argsort(dists)
        candidates = candidates[order]
        
        for idx in order:
            if time.time() > deadline:
                break
            x = int(candidates[idx, 0]); y = int(candidates[idx, 1])
            
            if is_valid_incremental(best_set, (x, y)):
                best_set = best_set[:] + [(x, y)]
                n_base = len(best_set)
                break
    
    # Also try using rational parametrization for more targeted search
    if time.time() < deadline:
        # Try circles centered at origin with various radii
        for R in [35, 50, 65, 80, 100, 120, 150]:
            if time.time() > deadline:
                break
            cands = generate_candidates_on_circle(0, 0, R, 50)
            cands = [p for p in cands if p not in best_set]
            for x, y in cands:
                if time.time() > deadline:
                    break
                if is_valid_incremental(best_set, (x, y)):
                    best_set = best_set[:] + [(x, y)]
                    n_base = len(best_set)
                    break
    
    # Try rational points with small denominators
    if time.time() < deadline:
        for den in [2, 3, 4, 5, 6]:
            if time.time() > deadline:
                break
            for num_x in range(-100, 101):
                if time.time() > deadline:
                    break
                for num_y in range(-100, 101):
                    if time.time() > deadline:
                        break
                    x_frac = Fraction(num_x, den)
                    y_frac = Fraction(num_y, den)
                    if x_frac.denominator != 1 or y_frac.denominator != 1:
                        continue
                    x = x_frac.numerator
                    y = y_frac.numerator
                    if (x, y) in best_set:
                        continue
                    if is_valid_incremental(best_set, (x, y)):
                        best_set = best_set[:] + [(x, y)]
                        n_base = len(best_set)
                        break
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

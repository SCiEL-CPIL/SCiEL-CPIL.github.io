def construct():
        import math
        import time
        import numpy as np
        
        start_time = time.time()
        deadline = start_time + 140
        
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
        
        for search_range in [200, 400, 700, 1200, 2000, 3500, 6000]:
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
            
            for x, y in candidates:
                if time.time() > deadline:
                    break
                x = int(x); y = int(y)
                
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

def construct():
    import random
    import numpy as np
    
    N = 16
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = N * N
    
    def is_isosceles(a, b, c):
        d1 = (a[0]-b[0])**2 + (a[1]-b[1])**2
        d2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
        d3 = (b[0]-c[0])**2 + (b[1]-c[1])**2
        return d1 == d2 or d1 == d3 or d2 == d3
    
    def is_valid_no_isosceles(points):
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    if is_isosceles(points[i], points[j], points[k]):
                        return False
        return True
    
    def is_maximal(points, all_pts):
        point_set = set(points)
        for p in all_pts:
            if p in point_set:
                continue
            # Check if p can be added
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if is_isosceles(p, points[i], points[j]):
                        break
                else:
                    continue
                break
            else:
                return False  # p can be added, so not maximal
        return True
    
    best_set = None
    best_size = n_points + 1
    
    for trial in range(300):
        random.seed(trial)
        perm = all_points[:]
        random.shuffle(perm)
        
        chosen = []
        for p in perm:
            can_add = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if is_isosceles(p, chosen[i], chosen[j]):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                chosen.append(p)
        
        # Check maximality
        if not is_maximal(chosen, all_points):
            continue
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    return best_set if best_set else all_points

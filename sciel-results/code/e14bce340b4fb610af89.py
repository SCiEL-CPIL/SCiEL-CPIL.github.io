import random

N = 64

def construct():
    all_points = [(x, y) for y in range(N) for x in range(N)]
    best_points = []
    best_size = 0
    
    # We will try a few different strategies:
    # 1. Random restarts
    # 2. Maybe a specific pattern?
    
    for seed in range(100):
        random.seed(seed)
        # Shuffle a copy
        candidates = all_points[:]
        random.shuffle(candidates)
        
        selected = []
        pair_dists = set()
        
        for p in candidates:
            px, py = p
            
            # 1. Compute distances to all selected points
            d_p_to_s = []
            for a in selected:
                ax, ay = a
                d_sq = (px - ax)**2 + (py - ay)**2
                d_p_to_s.append(d_sq)
            
            # 2. Check if any two distances are equal (Isosceles with P as apex)
            if len(d_p_to_s) != len(set(d_p_to_s)):
                continue
            
            # 3. Check if any distance from P is equal to any existing pair distance
            #    (Isosceles with P as base vertex)
            #    This covers d(P,A) == d(A,B)
            has_conflict = False
            for d_sq in d_p_to_s:
                if d_sq in pair_dists:
                    has_conflict = True
                    break
            
            if has_conflict:
                continue
            
            # Valid
            selected.append(p)
            # Update pair_dists with new distances from P to existing points
            pair_dists.update(d_p_to_s)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_points = selected
            
    return best_points

import time
import random
import itertools

N = 20
DEADLINE = 140  # seconds

def _collinear3(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _collinear3(points[i], points[j], points[k]) and \
                       _collinear3(points[i], points[j], points[l]):
                        return True
    return False

def _max_no_three_collinear(points):
    n = len(points)
    # Precompute collision masks
    # conflict[i][j] is a bitmask of k such that i, j, k are collinear
    # Actually, we need: if we have chosen set C, and we want to add p,
    # we must ensure no two points in C form a triple with p.
    # And if we add p, we must remove from available any q such that
    # there exists r in C such that p, q, r are collinear.
    
    # Let's precompute for each pair (i, j), the set of k such that i, j, k are collinear.
    # triple_mask[i][j] = bitmask of k
    triple_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _collinear3(points[i], points[j], points[k]):
                    mask |= (1 << k)
            triple_mask[i][j] = mask
            triple_mask[j][i] = mask
            
    best = [0]
    
    def backtrack(available_mask, chosen_mask, count):
        if available_mask == 0:
            if count > best[0]:
                best[0] = count
            return
            
        # Upper bound
        if count + bin(available_mask).count('1') <= best[0]:
            return
            
        # Pick a point. Heuristic: pick the one with most conflicts in available?
        # Or just the first one.
        # To speed up, pick a point that is in many triples?
        # For now, pick the lowest set bit.
        p = (available_mask & -available_mask).bit_length() - 1
        
        # Option 1: Exclude p
        backtrack(available_mask & ~(1 << p), chosen_mask, count)
        
        # Option 2: Include p
        # Check if p can be added: no two points in chosen_mask form a triple with p.
        # i.e., for all r in chosen_mask, triple_mask[p][r] & chosen_mask == 0?
        # No, triple_mask[p][r] contains k. We need to check if there is ANY r, s in chosen_mask
        # such that s is in triple_mask[p][r].
        # Simpler: Iterate through r in chosen_mask. If triple_mask[p][r] & chosen_mask != 0, then conflict.
        can_add = True
        temp_chosen = chosen_mask
        while temp_chosen:
            r = (temp_chosen & -temp_chosen).bit_length() - 1
            temp_chosen &= temp_chosen - 1
            if triple_mask[p][r] & chosen_mask:
                can_add = False
                break
                
        if can_add:
            new_chosen = chosen_mask | (1 << p)
            # New available mask
            new_available = available_mask & ~(1 << p)
            
            # Prune available: remove any q such that there exists r in chosen_mask
            # such that p, q, r are collinear.
            # i.e., q is in triple_mask[p][r] for some r in chosen_mask.
            forbidden = 0
            temp_chosen = chosen_mask
            while temp_chosen:
                r = (temp_chosen & -temp_chosen).bit_length() - 1
                temp_chosen &= temp_chosen - 1
                forbidden |= triple_mask[p][r]
                
            new_available &= ~forbidden
            
            backtrack(new_available, new_chosen, count + 1)
            
    backtrack((1 << n) - 1, 0, 0)
    return best[0]

def _construct_seed():
    # Start with a 4x5 grid
    points = []
    for i in range(4):
        for j in range(5):
            points.append([i * 20, j * 20])
    return points

def _fix_validity(points, deadline):
    # Move points to break 4-collinearities
    # This is a simple repair step
    improved = True
    while improved and time.time() < deadline:
        improved = False
        if not _has_four_collinear(points):
            break
        # Find a 4-collinear set
        n = len(points)
        found = False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if _collinear3(points[i], points[j], points[k]) and \
                           _collinear3(points[i], points[j], points[l]):
                            # Move point l
                            orig = points[l][:]
                            for _ in range(1000):
                                new_x = random.randint(-100, 100)
                                new_y = random.randint(-100, 100)
                                if [new_x, new_y] in points:
                                    continue
                                points[l] = [new_x, new_y]
                                if not _has_four_collinear(points):
                                    found = True
                                    improved = True
                                    break
                            if not found:
                                points[l] = orig
                            break
                    if found: break
                if found: break
            if found: break
        if not found:
            # No 4-collinear found? Should not happen if _has_four_collinear returned True
            # Maybe it's a bug in _has_four_collinear or the move failed to fix it?
            # Actually, if _has_four_collinear is True, we should find one.
            # If we can't fix it, break to avoid infinite loop
            break
    return points

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate initial candidates
    candidates = []
    
    # 1. Perturbed Grid
    p1 = _construct_seed()
    p1 = _fix_validity(p1, deadline)
    candidates.append(p1)
    
    # 2. Random
    p2 = []
    while len(p2) < N:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if [x, y] not in p2:
            p2.append([x, y])
    p2 = _fix_validity(p2, deadline)
    candidates.append(p2)
    
    # Evaluate candidates
    best_score = float('inf')
    best_points = None
    
    for p in candidates:
        if time.time() > deadline:
            break
        score = _max_no_three_collinear(p)
        if score < best_score:
            best_score = score
            best_points = [pt[:] for pt in p]
            
    # Local Search
    if best_points and time.time() < deadline:
        current_points = [pt[:] for pt in best_points]
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                orig = current_points[i][:]
                for _ in range(50): # Try 50 random moves for each point
                    if time.time() > deadline:
                        break
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    if [new_x, new_y] in current_points:
                        continue
                    current_points[i] = [new_x, new_y]
                    
                    if _has_four_collinear(current_points):
                        current_points[i] = orig
                        continue
                        
                    new_score = _max_no_three_collinear(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Do not reset current_points, keep the improvement
                    else:
                        current_points[i] = orig
                        
        if current_score < best_score:
            best_score = current_score
            best_points = [pt[:] for pt in current_points]
            
    return [pt[:] for pt in best_points]

import time
import random

N = 16
TOTAL_POINTS = N * N

def _precompute_iso_masks():
    # dist2[i][j]
    dist2 = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    for i in range(TOTAL_POINTS):
        xi, yi = divmod(i, N)
        for j in range(TOTAL_POINTS):
            xj, yj = divmod(j, N)
            dist2[i][j] = (xi - xj)**2 + (yi - yj)**2
            
    iso_mask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    for i in range(TOTAL_POINTS):
        for j in range(i + 1, TOTAL_POINTS):
            mask = 0
            dij = dist2[i][j]
            # Optimization: Only check k != i, j
            for k in range(TOTAL_POINTS):
                if k == i or k == j:
                    continue
                dik = dist2[i][k]
                djk = dist2[j][k]
                if dij == dik or dij == djk or dik == djk:
                    mask |= (1 << k)
            iso_mask[i][j] = mask
            iso_mask[j][i] = mask
    return iso_mask

ISO_MASKS = _precompute_iso_masks()

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = None
    best_size = TOTAL_POINTS + 1
    
    # Run many restarts
    num_restarts = 200
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
            
        chosen = []
        available_mask = (1 << TOTAL_POINTS) - 1
        completed = True
        
        while available_mask:
            if time.time() > deadline:
                completed = False
                break
                
            best_p = -1
            max_blocked = -1
            best_union_mask = 0
            
            # Iterate available points
            avail = available_mask
            while avail:
                lsb = avail & -avail
                p = lsb.bit_length() - 1
                avail ^= lsb
                
                union_mask = 0
                for a in chosen:
                    union_mask |= ISO_MASKS[p][a]
                
                new_blocked = union_mask & available_mask
                count = new_blocked.bit_count()
                
                if count > max_blocked:
                    max_blocked = count
                    best_p = p
                    best_union_mask = new_blocked
                elif count == max_blocked:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_p = p
                        best_union_mask = new_blocked
            
            if best_p == -1:
                break
                
            chosen.append(best_p)
            available_mask &= ~(1 << best_p)
            available_mask &= ~best_union_mask
            
        if completed:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_set = chosen.copy()
                
    if best_set is None:
        # Fallback to seed logic (row-major greedy)
        chosen = []
        for p in range(TOTAL_POINTS):
            ok = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    # Check if p, chosen[i], chosen[j] is isosceles
                    # Use precomputed masks?
                    # p is new, so check if p is in ISO_MASKS[chosen[i]][chosen[j]]
                    if ISO_MASKS[chosen[i]][chosen[j]] & (1 << p):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(p)
        best_set = chosen
        
    # Convert indices to tuples
    return [(p // N, p % N) for p in best_set]

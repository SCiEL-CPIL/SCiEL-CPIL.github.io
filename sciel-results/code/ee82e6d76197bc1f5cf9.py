"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)

# Flatten pair mask for speed: index = u * V + v
# pm[u * V + v] is bitmask of w

def _check_valid_bitmask(x, y, z, pm):
    """
    Check if adding (x,y,z) creates a K(2,2,2).
    pm is a flat list of 256 ints.
    """
    # Indices for quick access
    xy = x * V + y
    # We need to check x', y', z'
    # Pre-calculate some masks? No, they depend on x', y'.
    
    used = {x, y, z}
    
    for xp in range(V):
        if xp in used:
            continue
        xpy = xp * V + y
        # Check if bit z is in pm[xpy] -> {xp, y, z} in S
        if not (pm[xpy] & (1 << z)):
            continue
            
        for yp in range(V):
            if yp in used or yp == xp:
                continue
            xyp = x * V + yp
            xyp_yp = xp * V + yp
            
            # Check if bit z is in pm[xyp] -> {x, yp, z} in S
            if not (pm[xyp] & (1 << z)):
                continue
            # Check if bit z is in pm[xyp_yp] -> {xp, yp, z} in S
            if not (pm[xyp_yp] & (1 << z)):
                continue
            
            # Now iterate zp
            for zp in range(V):
                if zp in used or zp == xp or zp == yp:
                    continue
                
                # Check remaining conditions involving zp
                # 1. {x, y, zp} in S -> bit zp in pm[xy]
                if not (pm[xy] & (1 << zp)):
                    continue
                # 3. {x, yp, zp} in S -> bit zp in pm[xyp]
                if not (pm[xyp] & (1 << zp)):
                    continue
                # 5. {xp, y, zp} in S -> bit zp in pm[xpy]
                if not (pm[xpy] & (1 << zp)):
                    continue
                # 7. {xp, yp, zp} in S -> bit zp in pm[xyp_yp]
                if not (pm[xyp_yp] & (1 << zp)):
                    continue
                
                return False
    return True

def _greedy_fast(order, deadline):
    pm = [0] * (V * V)
    trips_set = set()
    result = []
    
    for t in order:
        if time.time() >= deadline:
            break
        if t in trips_set:
            continue
        
        x, y, z = t
        if _check_valid_bitmask(x, y, z, pm):
            trips_set.add(t)
            result.append(t)
            # Update pm
            pm[x * V + y] |= (1 << z)
            pm[y * V + x] |= (1 << z)
            pm[x * V + z] |= (1 << y)
            pm[z * V + x] |= (1 << y)
            pm[y * V + z] |= (1 << x)
            pm[z * V + y] |= (1 << x)
            
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 148  # Tight margin
    
    best_result = None
    best_len = -1
    
    # We will run as many restarts as possible
    # Each restart is very fast now
    
    # Optimization: Use a local variable for range(V)
    range_V = range(V)
    
    # Pre-shuffle list of indices? No, shuffle the triples directly.
    
    # To maximize throughput, we minimize overhead in the loop.
    
    while time.time() < deadline:
        order = ALL_TRIPLES[:]
        random.shuffle(order)
        
        result = _greedy_fast(order, deadline)
        
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
            
        # Optional: If we have time, do a quick local search on the best result?
        # For now, rely on restarts.
        
    if best_result is None:
        return []
        
    return [list(t) for t in best_result]

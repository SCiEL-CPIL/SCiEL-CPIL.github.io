import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 128
    
    # Precompute all non-constant 4-term APs
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N):
            e = a + 3 * d
            if e <= N:
                aps.append((a, a + d, a + 2 * d, e))
    
    # For each position, which APs does it belong to
    pos_aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def is_valid_partial(coloring, pos, color):
        for ap_idx in pos_aps[pos]:
            ap = aps[ap_idx]
            all_assigned = True
            colors_in_ap = []
            for p in ap:
                if p == pos:
                    colors_in_ap.append(color)
                else:
                    if coloring[p - 1] is None:
                        all_assigned = False
                        break
                    colors_in_ap.append(coloring[p - 1])
            if all_assigned:
                if len(set(colors_in_ap)) < 3:
                    return False
        return True
    
    for K in range(3, 13):
        if time.time() > deadline:
            break
        
        for restart in range(500):
            if time.time() > deadline:
                break
            
            order = list(range(1, N + 1))
            random.shuffle(order)
            
            coloring = [None] * N
            valid = True
            
            for pos in order:
                colors_to_try = list(range(K))
                random.shuffle(colors_to_try)
                
                found = False
                for c in colors_to_try:
                    if is_valid_partial(coloring, pos, c):
                        coloring[pos - 1] = c
                        found = True
                        break
                
                if not found:
                    valid = False
                    break
            
            if valid:
                for ap in aps:
                    colors_in_ap = set(coloring[p - 1] for p in ap)
                    if len(colors_in_ap) < 3:
                        valid = False
                        break
                if valid:
                    return coloring
    
    # Fallback
    colors = []
    for x in range(1, N + 1):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    
    return colors

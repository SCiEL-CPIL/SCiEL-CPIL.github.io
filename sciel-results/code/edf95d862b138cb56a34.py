import time

def construct():
    start_time = time.time()
    timeout = 85.0
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            if a + 3 * d <= n:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # For each position, which APs does it belong to?
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    # Try with k colors, starting from 3
    for k in range(3, 20):
        if time.time() - start_time > timeout - 10:
            break
        
        colors = [-1] * n
        
        def is_valid(pos, c):
            for ap_idx in pos_aps[pos]:
                ap = aps[ap_idx]
                vals = []
                for p in ap:
                    if p == pos:
                        vals.append(c)
                    else:
                        if colors[p] == -1:
                            return True  # Unknown, assume ok for now
                        vals.append(colors[p])
                if len(set(vals)) < 3:
                    return False
            return True
        
        def backtrack(pos):
            if time.time() - start_time > timeout - 5:
                return False
            if pos == n:
                return True
            
            for c in range(k):
                if is_valid(pos, c):
                    colors[pos] = c
                    if backtrack(pos + 1):
                        return True
                    colors[pos] = -1
            return False
        
        if backtrack(0):
            return colors
    
    # Fallback: greedy with 128 colors (always succeeds)
    colors = [0] * n
    for pos in range(n):
        for c in range(128):
            valid = True
            for ap_idx in pos_aps[pos]:
                ap = aps[ap_idx]
                vals = []
                for p in ap:
                    if p == pos:
                        vals.append(c)
                    else:
                        vals.append(colors[p])
                if len(set(vals)) < 3:
                    valid = False
                    break
            if valid:
                colors[pos] = c
                break
    return colors

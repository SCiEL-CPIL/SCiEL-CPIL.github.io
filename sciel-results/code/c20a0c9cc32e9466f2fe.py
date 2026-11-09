"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    def check_full(coloring):
        for a in range(1, n + 1):
            for d in range(1, n):
                if a + 3 * d > n:
                    break
                if len({coloring[a-1], coloring[a+d-1], coloring[a+2*d-1], coloring[a+3*d-1]}) < 3:
                    return False
        return True
    
    # Precompute all 4-term APs for faster checking
    all_aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            if a + 3 * d > n:
                break
            all_aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    
    def check_full_fast(coloring):
        for ap in all_aps:
            if len({coloring[i] for i in ap}) < 3:
                return False
        return True
    
    # Start with identity coloring (valid, 128 colors)
    best = list(range(n))
    best_cnt = n
    
    # Local search: try to reduce number of colors
    current = list(range(n))
    used_colors = set(current)
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        pos = random.randint(0, n - 1)
        old = current[pos]
        
        # Try to assign a color already in use to reduce total
        if len(used_colors) > 3:
            candidates = list(used_colors)
            new = random.choice(candidates)
        else:
            new = random.randint(0, 127)
        
        if new == old:
            continue
            
        current[pos] = new
        if check_full_fast(current):
            cnt = len(set(current))
            if cnt < best_cnt:
                best_cnt = cnt
                best = current[:]
                used_colors = set(current)
        else:
            current[pos] = old
    
    return best
# EVOLVE-BLOCK-END

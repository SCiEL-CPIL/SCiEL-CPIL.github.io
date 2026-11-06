"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _build_aps():
    aps = []
    for start in range(N):
        for step in range(1, N):
            pos = list(range(start, N, step))
            if len(pos) >= 2:
                aps.append(pos)
    pos_to_aps = [[] for _ in range(N)]
    for i, pos in enumerate(aps):
        for p in pos:
            pos_to_aps[p].append(i)
    return aps, pos_to_aps

def _thue_morse():
    signs = np.zeros(N, dtype=np.int32)
    for i in range(N):
        signs[i] = 1 if bin(i).count('1') % 2 == 0 else -1
    return signs

def _local_search(signs, aps, pos_to_aps, deadline):
    signs = signs.copy()
    sums = np.array([int(signs[pos].sum()) for pos in aps], dtype=np.int32)
    current_max = int(np.max(np.abs(sums)))
    best_signs = signs.copy()
    best_max = current_max
    
    no_improve_count = 0
    for it in range(500000):
        if time.time() > deadline:
            break
        
        improved = False
        # Try all positions
        for p in range(N):
            old_sign = signs[p]
            new_sign = -old_sign
            delta = 2 * new_sign
            affected = pos_to_aps[p]
            if not affected:
                continue
            old_vals = sums[affected].copy()
            sums[affected] += delta
            new_max = int(np.max(np.abs(sums)))
            if new_max < current_max:
                signs[p] = new_sign
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                    no_improve_count = 0
                improved = True
                break
            else:
                sums[affected] = old_vals
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 100:
                break
        else:
            no_improve_count = 0
    
    return best_signs, best_max

def construct():
    start_time = time.time()
    deadline = start_time + 78.0
    
    aps, pos_to_aps = _build_aps()
    
    signs = _thue_morse()
    signs, best_max = _local_search(signs, aps, pos_to_aps, deadline)
    
    rng = random.Random(7)
    while time.time() < deadline - 2.0:
        signs_r = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int32)
        signs_r, rmax = _local_search(signs_r, aps, pos_to_aps, deadline)
        if rmax < best_max:
            best_max = rmax
            signs = signs_r
    
    return [int(x) for x in signs]

# EVOLVE-BLOCK-END

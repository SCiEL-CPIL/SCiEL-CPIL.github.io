import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3
    
    def to_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb
        
    def from_idx(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb

    n_pts = q * q # 2401
    
    # Directions
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))
    dirs.append((0, 0, 0, 1))
    n_dirs = len(dirs) # 50
    
    def field_mul(a, b, c, d):
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)
        
    def field_add(a, b, c, d):
        return ((a + c) % p, (b + d) % p)

    # Precompute neighbors
    neighbors = [[[] for _ in range(n_dirs)] for _ in range(n_pts)]
    
    for di, (vxa, vxb, vya, vyb) in enumerate(dirs):
        for pi in range(n_pts):
            xa, xb, ya, yb = from_idx(pi)
            for t in range(1, q):
                ta = t // p
                tb = t % p
                nxa, nxb = field_mul(ta, tb, vxa, vxb)
                nya, nyb = field_mul(ta, tb, vya, vyb)
                nxa, nxb = field_add(xa, xb, nxa, nxb)
                nya, nyb = field_add(ya, yb, nya, nyb)
                idx = to_idx(nxa, nxb, nya, nyb)
                neighbors[pi][di].append(idx)
                
    # State
    in_S = np.zeros(n_pts, dtype=bool)
    free_mask = np.zeros(n_pts, dtype=np.uint64)
    free_count = np.zeros(n_pts, dtype=np.int32)
    
    full_mask = (1 << n_dirs) - 1
    
    def try_add(x):
        if in_S[x] or free_count[x] == 0:
            return False
        for v in range(n_dirs):
            mask_v = 1 << v
            for y in neighbors[x][v]:
                if free_count[y] == 1 and (free_mask[y] & mask_v):
                    return False
        return True

    def add(x):
        in_S[x] = True
        for v in range(n_dirs):
            mask_v = 1 << v
            for y in neighbors[x][v]:
                free_mask[y] &= ~mask_v
                free_count[y] = free_mask[y].bit_count()
                
    def remove(x):
        in_S[x] = False
        for v in range(n_dirs):
            mask_v = 1 << v
            for y in neighbors[x][v]:
                free_mask[y] |= mask_v
                free_count[y] = free_mask[y].bit_count()
                
    best_size = 0
    best_in_S = np.zeros(n_pts, dtype=bool)
    
    # Random Restart Local Search
    while time.time() < deadline - 10:
        # Reset state
        in_S[:] = False
        free_mask[:] = full_mask
        free_count[:] = n_dirs
        
        # Greedy Phase
        order = list(range(n_pts))
        random.shuffle(order)
        
        for x in order:
            if try_add(x):
                add(x)
                
        current_size = in_S.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_S = in_S.copy()
            
        # Local Search Phase
        # Try to improve by removing and re-adding
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Try to add a random point
            candidates = [i for i in range(n_pts) if not in_S[i] and free_count[i] > 0]
            if candidates:
                x = random.choice(candidates)
                if try_add(x):
                    add(x)
                    if in_S.sum() > best_size:
                        best_size = in_S.sum()
                        best_in_S = in_S.copy()
            else:
                # No point can be added. Remove a random point from S to unblock
                s_points = np.where(in_S)[0]
                if len(s_points) > 0:
                    x = random.choice(s_points)
                    remove(x)
                    
    # Build output: N is complement of S
    points = []
    for pi in range(n_pts):
        if not best_in_S[pi]:
            xa, xb, ya, yb = from_idx(pi)
            points.append([[xa, xb], [ya, yb]])
            
    if not points:
        for pi in range(n_pts):
            xa, xb, ya, yb = from_idx(pi)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    Q = q * q
    k_const = 3 # alpha^2 = 3
    
    # Generate all points as flat indices 0..2400
    # Point index i corresponds to (xa, xb, ya, yb)
    # i = ((xa * p + xb) * p + ya) * p + yb
    
    # Precompute direction for each pair (x, y)
    # Directions are defined by vectors v = (1, m) for m in F_q9 and (0, 1)
    # We need to map difference delta = y - x to a direction index.
    
    # Field operations in F_49
    def to_field(i):
        # i is 0..48
        a = i % p
        b = i // p
        return (a, b)
        
    def from_field(a, b):
        return a + b * p
        
    def field_add(a, b):
        a0, a1 = to_field(a)
        b0, b1 = to_field(b)
        return from_field((a0+b0)%p, (a1+b1)%p)
        
    def field_sub(a, b):
        a0, a1 = to_field(a)
        b0, b1 = to_field(b)
        return from_field((a0-b0)%p, (a1-b1)%p)
        
    def field_mul(a, b):
        a0, a1 = to_field(a)
        b0, b1 = to_field(b)
        # (a0 + a1 alpha) * (b0 + b1 alpha) = a0 b0 + a0 b1 alpha + a1 b0 alpha + a1 b1 alpha^2
        # = (a0 b0 + a1 b1 * k_const) + (a0 b1 + a1 b0) alpha
        r0 = (a0 * b0 + a1 * b1 * k_const) % p
        r1 = (a0 * b1 + a1 * b0) % p
        return from_field(r0, r1)
        
    # Define the 50 direction vectors
    # 49 directions (1, m) for m in F_49
    # 1 direction (0, 1)
    dirs = []
    for m in range(q):
        dirs.append((1, m))
    dirs.append((0, 1))
    
    # Precompute delta -> dir_index
    # delta is a point in F_49^2, represented as (dx, dy) where dx, dy in F_49
    # We need to check if (dx, dy) is a multiple of v = (vx, vy)
    # (dx, dy) = t * (vx, vy)
    # If vx != 0, t = dx / vx. Check if dy == t * vy.
    # If vx == 0, then dx must be 0. t = dy / vy.
    
    delta_to_dir = {}
    for dx in range(q):
        for dy in range(q):
            if dx == 0 and dy == 0:
                continue
            for d_idx, (vx, vy) in enumerate(dirs):
                if vx == 0:
                    if dx == 0:
                        # t = dy / vy
                        # vy is 1, so t = dy
                        # Check if (dx, dy) = t * (0, 1) => (0, t). Yes.
                        delta_to_dir[(dx, dy)] = d_idx
                        break
                    else:
                        continue
                else:
                    # vx = 1
                    # t = dx
                    # Check if dy == t * vy = dx * vy
                    if field_mul(dx, vy) == dy:
                        delta_to_dir[(dx, dy)] = d_idx
                        break
            
    # Helper to get point coords from flat index
    def get_coords(i):
        yb = i % p
        ya = (i // p) % p
        xb = (i // (p*p)) % p
        xa = (i // (p*p*p)) % p
        return (xa, xb, ya, yb)
        
    def get_field_coords(i):
        xa, xb, ya, yb = get_coords(i)
        x = xa + xb * p
        y = ya + yb * p
        return x, y
        
    def get_delta_dir(i, j):
        x1, y1 = get_field_coords(i)
        x2, y2 = get_field_coords(j)
        dx = field_sub(x2, x1)
        dy = field_sub(y2, y1)
        return delta_to_dir[(dx, dy)]
        
    # Greedy construction of K
    best_K_size = 0
    best_K = set()
    
    for seed in range(20):
        random.seed(seed)
        K = set()
        blocked = np.zeros((Q, 50), dtype=bool)
        count_blocked = np.zeros(Q, dtype=int)
        
        candidates = list(range(Q))
        random.shuffle(candidates)
        
        for y in candidates:
            if y in K:
                continue
                
            # Check if adding y invalidates any x
            valid = True
            affected = []
            
            for x in range(Q):
                if x == y:
                    continue
                d = get_delta_dir(x, y)
                if not blocked[x, d]:
                    # This direction becomes blocked for x
                    affected.append((x, d))
                    # Check if x becomes fully blocked
                    # We need to simulate the count
                    # count_blocked[x] + 1 == 50?
                    # We don't have current count in a variable we can easily peek without updating?
                    # Let's just update and check.
                    pass
            
            # To be efficient, we should check BEFORE updating.
            # But we need to know which (x, d) are newly blocked.
            
            # Re-do:
            affected = []
            for x in range(Q):
                if x == y:
                    continue
                d = get_delta_dir(x, y)
                if not blocked[x, d]:
                    affected.append((x, d))
                    
            # Check if any x in affected would reach 50
            for x, d in affected:
                if count_blocked[x] + 1 == 50:
                    valid = False
                    break
            
            if valid:
                # Commit
                for x, d in affected:
                    blocked[x, d] = True
                    count_blocked[x] += 1
                K.add(y)
            else:
                # Skip y
                pass
                
        if len(K) > best_K_size:
            best_K_size = len(K)
            best_K = K.copy()
            
    # Construct N as complement of best_K
    N = []
    for i in range(Q):
        if i not in best_K:
            xa, xb, ya, yb = get_coords(i)
            N.append([[xa, xb], [ya, yb]])
            
    return np.array(N, dtype=np.int64)

# EVOLVE-BLOCK-END

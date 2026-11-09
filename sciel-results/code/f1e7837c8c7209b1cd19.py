"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools

num_points = N * N
coords = [(i // N, i % N) for i in range(num_points)]

# Precompute squared distances
dist = [[0] * num_points for _ in range(num_points)]
for i in range(num_points):
    x1, y1 = coords[i]
    for j in range(i + 1, num_points):
        x2, y2 = coords[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

# Precompute forbid masks: forbid[a][b] has bit p set iff {p,a,b} is isosceles
forbid = [[0] * num_points for _ in range(num_points)]
for a in range(num_points):
    for b in range(a + 1, num_points):
        d_ab = dist[a][b]
        mask = (1 << a) | (1 << b)  # a and b themselves
        for p in range(num_points):
            if p == a or p == b:
                continue
            d_pa = dist[p][a]
            d_pb = dist[p][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                mask |= (1 << p)
        forbid[a][b] = mask
        forbid[b][a] = mask

all_points_mask = (1 << num_points) - 1

def compute_uf(S_list):
    """Compute union of forbid masks for all pairs in S_list."""
    uf = 0
    n = len(S_list)
    for i in range(n):
        si = S_list[i]
        for j in range(i + 1, n):
            uf |= forbid[si][S_list[j]]
    return uf

def construct():
    start_time = time.time()
    time_limit = 320  # 10s safety margin
    
    best_size = num_points + 1
    best_set = None
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # Random greedy initialization
        order = list(range(num_points))
        random.shuffle(order)
        
        current_set = []
        current_mask = 0
        union_forbid = 0
        
        for p in order:
            if not (union_forbid & (1 << p)):
                for q in current_set:
                    union_forbid |= forbid[p][q]
                current_set.append(p)
                current_mask |= (1 << p)
        
        # Repair to maximal: add any addable point
        changed = True
        while changed:
            changed = False
            for p in range(num_points):
                if not (current_mask & (1 << p)):
                    if not (union_forbid & (1 << p)):
                        for q in current_set:
                            union_forbid |= forbid[p][q]
                        current_set.append(p)
                        current_mask |= (1 << p)
                        changed = True
        
        # Local search: try removing points to shrink while staying maximal
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            s_list = list(current_set)
            random.shuffle(s_list)
            
            for p in s_list:
                if time.time() - start_time > time_limit:
                    break
                new_set = [q for q in current_set if q != p]
                new_mask = current_mask & ~(1 << p)
                new_uf = compute_uf(new_set)
                if (new_uf | new_mask) == all_points_mask:
                    current_set = new_set
                    current_mask = new_mask
                    union_forbid = new_uf
                    improved = True
                    break
        
        # Update best
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
        
        # Early termination for very small sets
        if best_size <= 5:
            break
    
    if best_set is None:
        return []
    return [coords[i] for i in best_set]

# EVOLVE-BLOCK-END

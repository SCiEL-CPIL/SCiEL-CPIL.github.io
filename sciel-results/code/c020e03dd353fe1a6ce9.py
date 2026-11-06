import numpy as np
import time
import itertools
import random

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)
coords = np.array(POINTS)

# Precompute squared distances
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

# Precompute Pairs[p]: list of (u, v) such that (p, u, v) is isosceles
# u < v, u != p, v != p
Pairs = [[] for _ in range(N_POINTS)]

for p in range(N_POINTS):
    for u in range(N_POINTS):
        if u == p:
            continue
        for v in range(u + 1, N_POINTS):
            if v == p:
                continue
            # Check if (p, u, v) is isosceles
            dp_u = D[p, u]
            dp_v = D[p, v]
            du_v = D[u, v]
            if dp_u == dp_v or dp_u == du_v or dp_v == du_v:
                Pairs[p].append((u, v))

def is_blocked(p, S_mask):
    """
    Check if point p is blocked by set S.
    p is blocked if there exist u, v in S such that (p, u, v) is isosceles.
    """
    for u, v in Pairs[p]:
        if ((S_mask >> u) & 1) and ((S_mask >> v) & 1):
            return True
    return False

def is_maximal(S_mask):
    """
    Check if set S is maximal.
    """
    # Check all points not in S
    for p in range(N_POINTS):
        if not ((S_mask >> p) & 1):
            if not is_blocked(p, S_mask):
                return False
    return True

def can_add(p, S_mask):
    """
    Check if point p can be added to S without creating an isosceles triangle.
    This is equivalent to: no u, v in S such that (p, u, v) is isosceles.
    """
    for u, v in Pairs[p]:
        if ((S_mask >> u) & 1) and ((S_mask >> v) & 1):
            return False
    return True

def popcount(x):
    return bin(x).count('1')

def greedy_maximal():
    S_mask = 0
    perm = list(range(N_POINTS))
    random.shuffle(perm)
    for p in perm:
        if can_add(p, S_mask):
            S_mask |= (1 << p)
    return S_mask

def construct():
    start_time = time.time()
    deadline = 290  # Safety margin
    
    best_mask = greedy_maximal()
    best_size = popcount(best_mask)
    
    current_mask = best_mask
    T = 1.0
    
    # Precompute list of all points for random selection
    all_points = list(range(N_POINTS))
    
    while time.time() - start_time < deadline:
        # Get points in and not in current set
        points_in = [i for i in all_points if (current_mask >> i) & 1]
        points_out = [i for i in all_points if not ((current_mask >> i) & 1)]
        
        if not points_in or not points_out:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try Removal
            u = random.choice(points_in)
            S_temp = current_mask & ~(1 << u)
            
            if is_maximal(S_temp):
                # Direct improvement
                current_mask = S_temp
                size = popcount(current_mask)
                if size < best_size:
                    best_size = size
                    best_mask = current_mask
                    T = 1.0
            else:
                # Try Repair: Add one point to S_temp to make it maximal
                # Find unblocked points in S_temp
                unblocked = []
                for p in points_out:
                    if not is_blocked(p, S_temp):
                        unblocked.append(p)
                
                if unblocked:
                    # Try a few random candidates from unblocked
                    candidates = random.sample(unblocked, min(len(unblocked), 10))
                    for a in candidates:
                        if can_add(a, S_temp):
                            S_try = S_temp | (1 << a)
                            if is_maximal(S_try):
                                # Swap move
                                if random.random() < T:
                                    current_mask = S_try
                                    size = popcount(current_mask)
                                    if size < best_size:
                                        best_size = size
                                        best_mask = current_mask
                                break # Only try one repair per removal attempt
        else:
            # Try Swap: Remove u, Add v
            u = random.choice(points_in)
            v = random.choice(points_out)
            S_temp = current_mask & ~(1 << u)
            
            if can_add(v, S_temp):
                S_try = S_temp | (1 << v)
                if is_maximal(S_try):
                    # Accept with probability T
                    if random.random() < T:
                        current_mask = S_try
                        size = popcount(current_mask)
                        if size < best_size:
                            best_size = size
                            best_mask = current_mask
        
        # Cool down
        T *= 0.999
        
    # Convert best_mask to list of points
    result = [POINTS[i] for i in range(N_POINTS) if (best_mask >> i) & 1]
    return result

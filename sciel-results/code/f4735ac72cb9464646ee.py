import numpy as np
import time
import random

N = 16
N2 = N * N
DEADLINE = time.time() + 290

# Precompute coordinates
coords = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distance matrix
xs = [c[0] for c in coords]
ys = [c[1] for c in coords]
xs_arr = np.array(xs)
ys_arr = np.array(ys)

dx = xs_arr[:, np.newaxis] - xs_arr[np.newaxis, :]
dy = ys_arr[:, np.newaxis] - ys_arr[np.newaxis, :]
dist_matrix = dx**2 + dy**2

def can_add(p, S_list):
    if not S_list:
        return True
    dists = dist_matrix[p, S_list]
    # Check for duplicate distances
    # Using set is efficient for small lists
    return len(set(dists)) == len(dists)

def get_addable_points(S_set, S_list):
    addable = []
    for p in range(N2):
        if p in S_set:
            continue
        if can_add(p, S_list):
            addable.append(p)
    return addable

def make_maximal(S_list, rng):
    S_set = set(S_list)
    S_list = list(S_list)
    
    while True:
        addable = get_addable_points(S_set, S_list)
        if not addable:
            break
        
        # Heuristic: Pick a point that might block others?
        # For now, random choice. Local search will refine.
        p = rng.choice(addable)
        S_list.append(p)
        S_set.add(p)
        
    return S_list

def greedy_construct(seed_val):
    rng = random.Random(seed_val)
    order = list(range(N2))
    rng.shuffle(order)
    
    S_list = []
    S_set = set()
    
    for p in order:
        if can_add(p, S_list):
            S_list.append(p)
            S_set.add(p)
    
    # Ensure maximal
    S_list = make_maximal(S_list, rng)
    return S_list

def local_search(S_initial, rng, max_iters=500):
    S_list = list(S_initial)
    best_list = list(S_initial)
    best_len = len(S_initial)
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        if len(S_list) <= 1:
            break
            
        # Pick a random point to remove
        idx = rng.randint(0, len(S_list) - 1)
        p_remove = S_list[idx]
        
        # Create S without p_remove
        S_temp = [x for i, x in enumerate(S_list) if i != idx]
        
        # Repair to make maximal
        S_repaired = make_maximal(S_temp, rng)
        
        if len(S_repaired) < best_len:
            best_list = S_repaired
            best_len = len(S_repaired)
            S_list = S_repaired
        elif len(S_repaired) == best_len:
            # Accept with some probability to explore
            if rng.random() < 0.2:
                S_list = S_repaired
        else:
            # Revert to best
            S_list = list(best_list)
            
    return best_list

def construct():
    best_solution = None
    best_len = float('inf')
    
    # Try multiple random restarts
    seed_counter = 0
    while time.time() < DEADLINE - 10:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # 1. Greedy construction
        S_greedy = greedy_construct(seed_counter)
        
        if len(S_greedy) < best_len:
            best_solution = S_greedy
            best_len = len(S_greedy)
            
        # 2. Local search improvement
        S_improved = local_search(S_greedy, rng, max_iters=200)
        
        if len(S_improved) < best_len:
            best_solution = S_improved
            best_len = len(S_improved)
            
        # Check time
        if time.time() > DEADLINE - 5:
            break
            
    if best_solution is None:
        # Fallback
        best_solution = greedy_construct(0)
        
    # Convert indices to coordinates
    return [coords[i] for i in best_solution]

import time
import random
from math import gcd
from itertools import combinations

# Precomputation
N = 128
lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        lcm_table[i][j] = i * j // gcd(i, j)

# conflicts[a][b] is a set of x such that {a, b, x} is a bad triple
# bad_pairs[x] is a list of (a, b) such that {a, b, x} is a bad triple
bad_pairs = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        L_ab = lcm_table[a][b]
        for x in range(1, N + 1):
            if x == a or x == b:
                continue
            if lcm_table[a][x] == L_ab and lcm_table[b][x] == L_ab:
                bad_pairs[x].append((a, b))
                # We don't need to add to a set, list is fine for iteration
                # Note: We only add (a,b) to bad_pairs[x] once for each pair.

def is_valid_addition_bitmask(x, mask):
    # Check if adding x to set represented by mask creates a bad triple
    # A bad triple involves x and two elements a, b from the set
    for a, b in bad_pairs[x]:
        # Check if a and b are in mask
        # Bit 0 corresponds to 1, Bit 1 to 2, etc.
        if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s safety margin
    
    best_size = 0
    best_mask = 0
    
    # Multi-start greedy
    base = list(range(1, N + 1))
    
    # Orderings
    orders = [base]
    random.seed(42)
    for _ in range(10):
        shuffled = base.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    # Heuristic: Sort by degree in conflict hypergraph?
    # Degree of x is len(bad_pairs[x])
    # Picking low degree first is a common heuristic for packing
    sorted_by_degree = sorted(base, key=lambda x: len(bad_pairs[x]))
    orders.append(sorted_by_degree)
    orders.append(sorted(base, key=lambda x: len(bad_pairs[x]), reverse=True))

    for order in orders:
        if time.time() > deadline:
            break
            
        current_mask = 0
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition_bitmask(x, current_mask):
                current_mask |= (1 << (x - 1))
        
        current_size = bin(current_mask).count('1')
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
    # Local Search / Simulated Annealing on the best mask
    if time.time() < deadline:
        current_mask = best_mask
        current_size = best_size
        
        # Precompute list of vertices not in mask for faster sampling
        # We will use random restarts of local search
        
        def get_vertices(mask):
            return [i+1 for i in range(N) if (mask >> i) & 1]
        
        def get_complement_vertices(mask):
            return [i+1 for i in range(N) if not ((mask >> i) & 1)]
        
        # Intensive Local Search
        # Try to improve by:
        # 1. Adding any valid vertex (greedy fill)
        # 2. Swapping: Remove u, Add v (if valid), then try to add more.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Greedy Fill
            added = False
            for x in range(1, N + 1):
                if not ((current_mask >> (x - 1)) & 1):
                    if is_valid_addition_bitmask(x, current_mask):
                        current_mask |= (1 << (x - 1))
                        current_size += 1
                        added = True
                        improved = True
                        break # Restart greedy fill
            
            if not added:
                # 2. Swap Move: Remove a random vertex u, try to add a random vertex v
                # To escape local optima, we accept size-decreasing or size-neutral moves?
                # For now, let's try 1-remove, 1-add that results in same size, 
                # but potentially allows further additions.
                
                # Pick a random vertex to remove
                verts = get_vertices(current_mask)
                if not verts:
                    break
                u = random.choice(verts)
                
                # Try to find a v that can replace u or just be added after u is removed
                # Actually, if we remove u, we might be able to add multiple vertices.
                # Let's remove u, then greedily fill.
                
                trial_mask = current_mask & ~(1 << (u - 1))
                trial_size = current_size - 1
                
                # Greedily add any valid vertex to trial_mask
                for x in range(1, N + 1):
                    if not ((trial_mask >> (x - 1)) & 1):
                        if is_valid_addition_bitmask(x, trial_mask):
                            trial_mask |= (1 << (x - 1))
                            trial_size += 1
                            
                if trial_size > current_size:
                    current_mask = trial_mask
                    current_size = trial_size
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                else:
                    # If no improvement, we might be stuck.
                    # Try a different u?
                    # For efficiency, we break and rely on restarts or accept local max.
                    # Let's try a few random u's before giving up on this state.
                    pass
                    
        # If we have time, do random restarts of the whole process with the best mask as a seed?
        # Or just more local search iterations.
        
    return [i+1 for i in range(N) if (best_mask >> i) & 1]

if not construct():
    construct = lambda: [1]

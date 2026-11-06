import time
import random
import itertools

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)
ALL_MASK = (1 << NUM_POINTS) - 1

# Precompute conflict masks
# conflict_mask[i][j] is a bitmask of points p such that (p, grid[i], grid[j]) is isosceles
conflict_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

def dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def is_isosceles(a, b, c):
    d1 = dist_sq(a, b)
    d2 = dist_sq(a, c)
    d3 = dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

# Precomputation
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        a = GRID_POINTS[i]
        b = GRID_POINTS[j]
        mask = 0
        for k in range(NUM_POINTS):
            c = GRID_POINTS[k]
            if is_isosceles(c, a, b):
                mask |= (1 << k)
        conflict_mask[i][j] = mask
        conflict_mask[j][i] = mask

def get_union_mask(S_indices):
    """Compute the union of conflict masks for all pairs in S."""
    union = 0
    for i, j in itertools.combinations(S_indices, 2):
        union |= conflict_mask[i][j]
    return union

def is_maximal(S_indices):
    """Check if S is maximal using bitmask logic."""
    if len(S_indices) < 2:
        # If S has 0 or 1 points, it is never maximal unless grid is tiny?
        # Actually, if S is empty, any point can be added. So not maximal.
        # If S has 1 point, any other point can be added (no triangles). So not maximal.
        return False
        
    S_mask = 0
    for idx in S_indices:
        S_mask |= (1 << idx)
        
    forbidden_union = get_union_mask(S_indices)
    
    # Points that can be added are those in ALL_MASK but not in S_mask and not in forbidden_union
    # If there are no such points, S is maximal.
    available = ALL_MASK & ~(S_mask | forbidden_union)
    return available == 0

def greedy_construct(order_indices):
    """Construct a maximal set using the given order of point indices."""
    S_indices = []
    # We can maintain the union mask incrementally? 
    # No, because adding a point changes the pairs.
    # But for greedy, we just check if the new point is in the current forbidden union.
    current_union = 0
    S_set = set()
    
    for idx in order_indices:
        # Check if idx is in current_union
        if (current_union >> idx) & 1:
            continue
        
        # It is safe to add.
        # Update current_union: 
        # The new pairs are (idx, s) for all s in S_indices
        # So we need to OR conflict_mask[idx][s] for all s in S_indices
        for s in S_indices:
            current_union |= conflict_mask[idx][s]
            
        S_indices.append(idx)
        S_set.add(idx)
        
    return S_indices

def local_search_reduce(S_indices):
    """Try to remove points from S while maintaining maximality."""
    S_indices = list(S_indices)
    improved = True
    while improved:
        improved = False
        indices = list(range(len(S_indices)))
        random.shuffle(indices)
        for i in indices:
            p_idx = S_indices[i]
            S_new = S_indices[:i] + S_indices[i+1:]
            if is_maximal(S_new):
                S_indices = S_new
                improved = True
                break
    return S_indices

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set_indices = None
    best_size = float('inf')
    
    # Initial greedy with row-major order
    initial_order = list(range(NUM_POINTS))
    S = greedy_construct(initial_order)
    S = local_search_reduce(S)
    if len(S) < best_size:
        best_size = len(S)
        best_set_indices = S
        
    # Iterated Local Search
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        S = greedy_construct(order)
        S = local_search_reduce(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set_indices = S
            
    # Convert indices back to (x, y) tuples
    return [GRID_POINTS[i] for i in best_set_indices]

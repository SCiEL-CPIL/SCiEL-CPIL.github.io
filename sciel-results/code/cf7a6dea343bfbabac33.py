import itertools
import random
import time
from collections import defaultdict

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
dist = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        dist[i][j] = d
        dist[j][i] = d

# Precompute pair indices
pair_to_idx = {}
idx_to_pair = []
idx = 0
for u in range(NUM_POINTS):
    for v in range(u+1, NUM_POINTS):
        pair_to_idx[(u, v)] = idx
        idx_to_pair.append((u, v))
        idx += 1
NUM_PAIRS = idx

# Precompute conflict_bits[x]
# conflict_bits[x] is a bitset of pairs that form an isosceles triangle with point x.
conflict_bits = [0] * NUM_POINTS
for x in range(NUM_POINTS):
    mask = 0
    for p_idx in range(NUM_PAIRS):
        u, v = idx_to_pair[p_idx]
        du = dist[x][u]
        dv = dist[x][v]
        uv = dist[u][v]
        if du == dv or du == uv or dv == uv:
            mask |= (1 << p_idx)
    conflict_bits[x] = mask

def get_pairs_bits(set_indices):
    """Compute the bitset of pairs in the set."""
    mask = 0
    m = len(set_indices)
    for i in range(m):
        u = set_indices[i]
        for j in range(i+1, m):
            v = set_indices[j]
            if u > v:
                u, v = v, u
            p_idx = pair_to_idx[(u, v)]
            mask |= (1 << p_idx)
    return mask

def can_add(p_idx, set_indices, set_pairs_bits=None):
    if set_pairs_bits is None:
        set_pairs_bits = get_pairs_bits(set_indices)
    return (conflict_bits[p_idx] & set_pairs_bits) == 0

def is_maximal(set_indices, set_pairs_bits=None):
    if set_pairs_bits is None:
        set_pairs_bits = get_pairs_bits(set_indices)
    set_set = set(set_indices)
    for p in range(NUM_POINTS):
        if p not in set_set:
            if can_add(p, set_indices, set_pairs_bits):
                return False
    return True

def greedy_construction(order):
    chosen = []
    chosen_bits = 0
    for p in order:
        if can_add(p, chosen, chosen_bits):
            chosen.append(p)
            for q in chosen[:-1]:
                u, v = p, q
                if u > v: u, v = v, u
                p_idx = pair_to_idx[(u, v)]
                chosen_bits |= (1 << p_idx)
    return chosen

def local_search(set_indices):
    current = list(set_indices)
    improved = True
    while improved:
        improved = False
        current_bits = get_pairs_bits(current)
        
        # 1-moves: Try removing a point
        for i in range(len(current)):
            p = current[i]
            new_set = current[:i] + current[i+1:]
            if is_maximal(new_set):
                current = new_set
                improved = True
                break
        
        if improved:
            continue
            
        # 2-moves: Try swapping a point
        if len(current) > 0:
            current_set = set(current)
            candidates_out = [p for p in range(NUM_POINTS) if p not in current_set]
            
            # Try a limited number of random swaps
            num_swaps_to_try = min(50, len(candidates_out) * len(current))
            for _ in range(num_swaps_to_try):
                if not candidates_out: break
                p_in = random.choice(current)
                p_out = random.choice(candidates_out)
                
                new_set = [x for x in current if x != p_in]
                new_set.append(p_out)
                
                reduced_set = [x for x in current if x != p_in]
                reduced_bits = get_pairs_bits(reduced_set)
                
                if can_add(p_out, reduced_set, reduced_bits):
                    if is_maximal(new_set):
                        current = new_set
                        improved = True
                        break
                        
    return current

def simulated_annealing(set_indices, start_time, deadline):
    """
    Use simulated annealing to escape local minima.
    Moves:
    1. Add a point (if valid, but not maximal yet? No, we must maintain validity).
       Actually, we must maintain the property: no isosceles triangles.
       So we can only add points that are compatible.
       And we must maintain maximality? No, the final result must be maximal.
       But intermediate states in SA don't have to be maximal, just valid (no isosceles).
       However, if we allow non-maximal states, we might get stuck in a small valid set
       that is not maximal and cannot be extended? No, any valid set can be extended to a maximal one.
       But we want to minimize the size of the *maximal* set.
       
       Strategy:
       State: A valid set S (no isosceles triangles).
       Move:
       - Add a point p if compatible with S.
       - Remove a point p from S.
       - Swap p in S with q not in S, if q is compatible with S\{p}.
       
       Objective: Minimize |S|, but S must be "close" to maximal.
       Actually, the score is only for the final returned set which MUST be maximal.
       So we can run SA to find a small valid set, and then check if it's maximal.
       If not, we can try to make it maximal? No, making it maximal requires adding points, increasing size.
       
       Better approach:
       We want a small MAXIMAL set.
       Let's define the state as a maximal set.
       Moves:
       - Remove p: If S\{p} is still maximal, accept (size decreases).
       - Swap p, q: If (S\{p}) U {q} is maximal, accept (size same, but might allow future removals).
       
       This is essentially the local search we already have, but with random acceptance of worse (same size) moves to escape local minima.
       Since removals strictly improve size, we always accept them.
       Swaps keep size same. We accept swaps with probability exp(-delta/T) where delta is 0?
       Actually, if size is same, delta is 0, so we always accept?
       If we always accept swaps, we might cycle.
       Let's add a penalty for "quality" of the maximal set?
       Or just use random walks in the space of maximal sets of the same size.
       
       Let's try:
       1. Always accept removals (size decreases).
       2. Accept swaps with probability 0.5 (random walk).
       3. Temperature cools down, reducing swap acceptance to avoid cycling?
       Actually, for a random walk to escape a local minimum (where no removal is possible),
       we need to perform swaps that don't immediately allow removal, but lead to a state where removal is possible.
       
       Let's implement a simple random walk over maximal sets of the current size.
    """
    current = list(set_indices)
    current_size = len(current)
    best_set = list(current)
    best_size = current_size
    
    T = 1.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline - 5.0:
        # Try a random move
        move_type = random.random()
        
        if move_type < 0.3 and len(current) > 0:
            # Try removal
            p = random.choice(current)
            new_set = [x for x in current if x != p]
            if is_maximal(new_set):
                current = new_set
                current_size = len(current)
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(current)
                # Reset temperature? Or just keep cooling.
                T = max(T, 0.5) # Reheat slightly on improvement
        else:
            # Try swap
            if len(current) == 0:
                continue
            current_set = set(current)
            candidates_out = [p for p in range(NUM_POINTS) if p not in current_set]
            if not candidates_out:
                continue
            
            p_in = random.choice(current)
            p_out = random.choice(candidates_out)
            
            new_set = [x for x in current if x != p_in]
            new_set.append(p_out)
            
            # Check if new_set is valid and maximal
            reduced_set = [x for x in current if x != p_in]
            reduced_bits = get_pairs_bits(reduced_set)
            
            if can_add(p_out, reduced_set, reduced_bits):
                if is_maximal(new_set):
                    # Accept swap
                    # Probability of acceptance: 1.0 (since size is same)
                    # To avoid cycling, we might want to be more selective, but random walk is okay.
                    current = new_set
                    # No size change, so no update to best_size unless we find a removal later
                    # Cool down
                    T *= cooling
                    if T < T_min:
                        T = T_min
                        # Maybe restart or do something else?
                        # For now, just keep going.
        
    # Final check: ensure best_set is maximal
    if not is_maximal(best_set):
        # Fallback to current if it's maximal
        if is_maximal(current):
            best_set = current
        else:
            # Should not happen, but just in case
            best_set = set_indices
            
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Randomized Greedy + Local Search
    restarts = 0
    while time.time() < deadline - 30.0:
        restarts += 1
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        s = greedy_construction(order)
        s = local_search(s)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            
    # Phase 2: Simulated Annealing / Random Walk on best_set
    if best_set is not None:
        # Run SA for remaining time
        sa_set = simulated_annealing(best_set, start_time, deadline)
        if len(sa_set) < best_size:
            best_size = len(sa_set)
            best_set = sa_set

    if best_set is None:
        s = greedy_construction(list(range(NUM_POINTS)))
        best_set = s

    return [POINTS[i] for i in best_set]

"""Improved program for the suite task: no four-cycle with 4 distinct colors."""
import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 82.0  # safety margin before 90s kill

# Precompute all 4-cycles: for each 4-vertex subset, the 3 cycles
def build_cycles():
    """Return list of cycles, each cycle is a list of 4 edge indices."""
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = idx
            idx += 1
    def eidx(i, j):
        if i > j:
            i, j = j, i
        return edge_idx[(i, j)]
    
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    cycles.append([eidx(a,b), eidx(b,c), eidx(c,d), eidx(d,a)])
                    cycles.append([eidx(a,b), eidx(b,d), eidx(d,c), eidx(c,a)])
                    cycles.append([eidx(a,c), eidx(c,b), eidx(b,d), eidx(d,a)])
    return cycles

_CYCLES = build_cycles()
NUM_CYCLES = len(_CYCLES)

# For each edge, precompute which cycles it belongs to
_edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(_CYCLES):
    for ei in cyc:
        _edge_to_cycles[ei].append(ci)

def count_distinct_colors(colors):
    return len(set(colors))

def initial_coloring():
    """Linear ordering: c(i,j) = j"""
    colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            colors[idx] = j
            idx += 1
    return colors

def full_validity_check(colors):
    """Check no 4-cycle has 4 distinct colors."""
    arr = np.array(colors, dtype=np.int32)
    for cyc in _CYCLES:
        c0, c1, c2, c3 = arr[cyc[0]], arr[cyc[1]], arr[cyc[2]], arr[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def local_search_sa(colors, max_time=80.0, seed=42):
    """Simulated annealing to maximize distinct colors while maintaining validity.
    
    Strategy: maintain a valid coloring. At each step, try changing one edge to a new color.
    Only accept if the result remains valid. Use SA to escape local optima by accepting
    moves that don't decrease distinct colors (neutral moves) with some probability.
    """
    start_time = time.time()
    rng = random.Random(seed)
    colors = list(colors)
    best_colors = list(colors)
    best_count = count_distinct_colors(colors)
    
    # Track color usage counts
    color_count = [0] * 120
    for c in colors:
        color_count[c] += 1
    distinct_count = sum(1 for c in color_count if c > 0)
    
    # Track for each cycle whether it's a rainbow
    is_rainbow = np.zeros(NUM_CYCLES, dtype=bool)
    for ci, cyc in enumerate(_CYCLES):
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            is_rainbow[ci] = True
    
    # Temperature schedule
    T = 1.0
    cooling = 0.9999
    min_T = 0.001
    
    iteration = 0
    no_improve = 0
    max_no_improve = 5000
    
    while True:
        if time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE:
            break
        
        iteration += 1
        
        # Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Choose a new color: prefer unused colors to increase distinct count
        used = set(colors)
        unused = [c for c in range(120) if c not in used]
        
        if unused and rng.random() < 0.7:
            new_color = rng.choice(unused)
        else:
            new_color = rng.randint(0, 119)
            if new_color == old_color:
                continue
        
        # Check validity: only cycles containing edge e need rechecking
        # First check if changing would create a rainbow in any affected cycle
        valid = True
        for ci in _edge_to_cycles[e]:
            cyc = _CYCLES[ci]
            # Get the other 3 colors
            others = [colors[cyc[i]] for i in range(4) if cyc[i] != e]
            # Check if new_color + others would form 4 distinct
            all_colors = set([new_color] + others)
            if len(all_colors) == 4:
                valid = False
                break
        
        if valid:
            # Accept the move
            colors[e] = new_color
            # Update color counts
            color_count[old_color] -= 1
            color_count[new_color] += 1
            if color_count[old_color] == 0:
                distinct_count -= 1
            if color_count[new_color] == 1:
                distinct_count += 1
            
            # Update rainbow status for affected cycles
            for ci in _edge_to_cycles[e]:
                cyc = _CYCLES[ci]
                c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                was_rainbow = is_rainbow[ci]
                is_rainbow[ci] = (len({c0, c1, c2, c3}) == 4)
                # Note: since we checked validity, this should be False
        
            if distinct_count > best_count:
                best_count = distinct_count
                best_colors = list(colors)
                no_improve = 0
            else:
                no_improve += 1
            
            # Cooling
            T = max(min_T, T * cooling)
        else:
            # Reject: don't change
            no_improve += 1
            # Occasionally reset temperature to escape local optima
            if no_improve > max_no_improve:
                T = 0.5
                no_improve = 0
        
        # Periodic full check for safety
        if iteration % 10000 == 0:
            if not full_validity_check(colors):
                # Revert to best
                colors = list(best_colors)
                color_count = [0] * 120
                for c in colors:
                    color_count[c] += 1
                distinct_count = sum(1 for c in color_count if c > 0)
                is_rainbow = np.zeros(NUM_CYCLES, dtype=bool)
                for ci, cyc in enumerate(_CYCLES):
                    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                    if len({c0, c1, c2, c3}) == 4:
                        is_rainbow[ci] = True
    
    return best_colors, best_count

def greedy_coloring():
    """Greedy approach: try to assign distinct colors to edges while maintaining validity."""
    colors = [0] * NUM_EDGES
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = idx
            idx += 1
    
    # Start with all color 0
    used_colors = {0}
    
    # Try to assign new colors greedily
    for target_color in range(1, 120):
        # Find an edge that can be recolored to target_color
        found = False
        for e in range(NUM_EDGES):
            if colors[e] == target_color:
                continue
            # Check if changing edge e to target_color maintains validity
            valid = True
            for ci in _edge_to_cycles[e]:
                cyc = _CYCLES[ci]
                others = [colors[cyc[i]] for i in range(4) if cyc[i] != e]
                if len(set([target_color] + others)) == 4:
                    valid = False
                    break
            if valid:
                colors[e] = target_color
                found = True
                break
        if not found:
            break
    
    return colors, count_distinct_colors(colors)

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Start with linear ordering
    colors = initial_coloring()
    best_colors = list(colors)
    best_count = count_distinct_colors(colors)
    
    # Run local search from initial
    if time.time() < DEADLINE - 10:
        result, count = local_search_sa(colors, max_time=30.0, seed=42)
        if count > best_count:
            best_count = count
            best_colors = result
    
    # Try greedy approach
    if time.time() < DEADLINE - 15:
        greedy_result, greedy_count = greedy_coloring()
        if greedy_count > best_count:
            best_count = greedy_count
            best_colors = list(greedy_result)
        # Also try local search from greedy
        if time.time() < DEADLINE - 10:
            result, count = local_search_sa(greedy_result, max_time=20.0, seed=123)
            if count > best_count:
                best_count = count
                best_colors = result
    
    # Multiple restarts with different seeds
    for seed in [1, 7, 13, 99, 256, 512, 1024, 2048, 4096, 8192]:
        if time.time() > DEADLINE - 5:
            break
        
        # Create a random valid starting point
        start = list(initial_coloring())
        # Perturb slightly
        for _ in range(20):
            e = random.randint(0, NUM_EDGES - 1)
            start[e] = random.randint(0, 119)
        
        # Make sure it's valid; if not, revert to initial
        if not full_validity_check(start):
            start = list(initial_coloring())
        
        # Run local search
        remaining = DEADLINE - 5 - (time.time() - start_time)
        if remaining < 2:
            break
        
        result, count = local_search_sa(start, max_time=min(remaining, 15.0), seed=seed)
        if count > best_count:
            best_count = count
            best_colors = result
    
    # Final verification
    if not full_validity_check(best_colors):
        # Fallback to initial
        best_colors = initial_coloring()
    
    return best_colors

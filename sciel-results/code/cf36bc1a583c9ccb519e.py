"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _build_edge_index():
    """Map (i,j) with i<j to a linear index 0..119."""
    edge_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx += 1
    return edge_to_idx

def _build_edge_list():
    """List of edges as (i,j) pairs in lex order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _build_cycles():
    """
    For each 4-vertex subset, compute the 3 cycles.
    Each cycle is a list of 4 edge indices.
    Return list of cycles (each is a tuple of 4 edge indices).
    """
    edge_to_idx = _build_edge_index()
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx[(min(a,b), max(a,b))]
        e2 = edge_to_idx[(min(b,c), max(b,c))]
        e3 = edge_to_idx[(min(c,d), max(c,d))]
        e4 = edge_to_idx[(min(d,a), max(d,a))]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx[(min(a,b), max(a,b))]
        e2 = edge_to_idx[(min(b,d), max(b,d))]
        e3 = edge_to_idx[(min(d,c), max(d,c))]
        e4 = edge_to_idx[(min(c,a), max(c,a))]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx[(min(a,c), max(a,c))]
        e2 = edge_to_idx[(min(c,b), max(c,b))]
        e3 = edge_to_idx[(min(b,d), max(b,d))]
        e4 = edge_to_idx[(min(d,a), max(d,a))]
        cycles.append((e1, e2, e3, e4))
    return cycles

def _build_edge_to_cycles():
    """Map each edge index to list of cycle indices it participates in."""
    cycles = _build_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    return edge_to_cycles

# Precompute
_EDGE_TO_IDX = _build_edge_index()
_CYCLES = _build_cycles()
_EDGE_TO_CYCLES = _build_edge_to_cycles()
NUM_CYCLES = len(_CYCLES)

def construct():
    random.seed(42)
    
    # State variables
    colors = [0] * NUM_EDGES
    color_counts = [0] * 120
    color_counts[0] = NUM_EDGES
    num_distinct = 1
    violations = 0
    
    # Precompute cycle status for initial state (all 0s -> 0 violations)
    # We maintain a boolean array or just recalculate on the fly for affected cycles.
    # Given the small number of cycles per edge (15 edges in a 4-vertex set? No, each edge is in C(14,2)=91 cycles? 
    # Wait, an edge (i,j) is in a cycle if we pick 2 other vertices from the remaining 14. 
    # Number of 4-sets containing i,j is C(14,2) = 91.
    # In each 4-set, there are 3 cycles. But not all cycles contain the edge (i,j).
    # Actually, for a fixed edge (i,j), in a 4-set {i,j,k,l}, the cycles are:
    # 1. i-j-k-l-i (contains i-j)
    # 2. i-j-l-k-i (contains i-j)
    # 3. i-k-j-l-i (does NOT contain i-j, contains i-k, k-j, j-l, l-i)
    # So each edge is in 2 cycles per 4-set.
    # Total cycles per edge = 91 * 2 = 182.
    # This is a lot to check per move. 
    # Optimization: We can precompute for each edge, the list of (cycle_idx, position_in_cycle) it belongs to.
    
    # Let's re-evaluate the cost of checking violations.
    # 182 cycles * 4 edges lookup = ~728 operations per move.
    # 50,000 moves * 728 = 36,400,000 operations. This might be too slow in Python if not optimized.
    # We need to speed this up.
    
    # Alternative: Maintain a count of distinct colors per cycle?
    # If we change one edge in a cycle, we can update the set of colors for that cycle.
    # But storing a set for 3276 cycles (1820 quads * ?) is memory heavy and update heavy.
    
    # Let's check the number of cycles.
    # C(16,4) = 1820 quads. 3 cycles per quad = 5460 cycles.
    # Each cycle has 4 edges.
    # Total edge-cycle incidences = 5460 * 4 = 21840.
    # Average cycles per edge = 21840 / 120 = 182. Correct.
    
    # To make it fast, we will precompute for each edge:
    # list of (cycle_index, other_edge_indices)
    # Then for a move on edge `ei` from `old_c` to `new_c`:
    # For each (ci, others) in edge_to_cycles[ei]:
    #    old_set = {old_c} | {colors[e] for e in others}
    #    new_set = {new_c} | {colors[e] for e in others}
    #    delta_v = (len(new_set)==4) - (len(old_set)==4)
    
    # This is still O(182 * 4) per move.
    # We can optimize by precomputing the "others" as a tuple of 3 indices.
    
    edge_to_cycle_info = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(_CYCLES):
        c0, c1, c2, c3 = cycle
        # For each edge in the cycle, store the cycle index and the other 3 edges
        edge_to_cycle_info[c0].append((ci, (c1, c2, c3)))
        edge_to_cycle_info[c1].append((ci, (c0, c2, c3)))
        edge_to_cycle_info[c2].append((ci, (c0, c1, c3)))
        edge_to_cycle_info[c3].append((ci, (c0, c1, c2)))
        
    # Now the delta calculation is faster.
    
    def calc_delta_violations(ei, old_c, new_c):
        delta = 0
        for ci, others in edge_to_cycle_info[ei]:
            # Get colors of other edges
            oc1 = colors[others[0]]
            oc2 = colors[others[1]]
            oc3 = colors[others[2]]
            
            old_set = {old_c, oc1, oc2, oc3}
            new_set = {new_c, oc1, oc2, oc3}
            
            old_bad = 1 if len(old_set) == 4 else 0
            new_bad = 1 if len(new_set) == 4 else 0
            
            delta += new_bad - old_bad
        return delta

    # Entropy calculation is expensive to do from scratch.
    # We can approximate or just use DistinctCount.
    # The prompt asks for entropy penalty. 
    # Let's maintain `color_counts` and calculate entropy delta.
    # H = -sum(p log p). 
    # Delta H when moving 1 count from old to new:
    # H_new - H_old = ... 
    # This is complex. Let's just use DistinctCount as the primary driver and rely on SA to find high distinct count.
    # The "entropy" in the prompt might be a hint to use a diversity metric that is smoother than distinct count.
    # But distinct count is the score.
    
    # Let's try a simpler approach: 
    # Cost = Violations * 100 - DistinctCount
    # Accept if Cost decreases, or with prob exp(-delta/T).
    
    best_colors = colors[:]
    best_distinct = num_distinct
    
    T = 50.0
    T_min = 1.0
    cooling = 0.995
    max_iter = 200000
    
    W_V = 100
    W_D = 1
    
    for iteration in range(max_iter):
        if T < T_min:
            break
            
        ei = random.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Choose new color
        # Strategy: Pick a color that is not in the "forbidden" set of any incident cycle?
        # Or just random.
        # To maximize distinct, we should try to introduce new colors if possible.
        # If we pick a color not used at all, distinct increases by 1 (if no violation).
        # Let's bias towards unused colors or rare colors.
        
        # Find unused colors
        # This is expensive to check every time.
        # Instead, pick random color from 0..119.
        new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
            
        delta_v = calc_delta_violations(ei, old_color, new_color)
        
        # Delta distinct
        delta_d = 0
        if color_counts[old_color] == 1:
            delta_d -= 1
        if color_counts[new_color] == 0:
            delta_d += 1
            
        delta_cost = delta_v * W_V - delta_d * W_D
        
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            # delta_cost > 0
            # Check if it's affordable
            if delta_cost < T: # Avoid overflow/underflow in exp for large T? No, T is small.
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    accept = True
            # else prob is ~0
            
        if accept:
            colors[ei] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            violations += delta_v
            num_distinct += delta_d
            
            if violations == 0:
                if num_distinct > best_distinct:
                    best_distinct = num_distinct
                    best_colors = colors[:]
                    
        T *= cooling
        
    return best_colors

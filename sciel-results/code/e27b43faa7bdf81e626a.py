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
    
    # Initialize with a valid low-diversity coloring to start from a feasible point
    # e.g., all 0s is valid (0 distinct colors? no, 1 distinct color). 
    # Actually, all same color means 0 violations.
    colors = [0] * NUM_EDGES
    color_counts = [0] * 120
    color_counts[0] = NUM_EDGES
    num_distinct = 1
    
    # Precompute violation counts for initial state (should be 0)
    violations = 0
    
    # To speed up delta calculation, we need to know current state of each cycle.
    # A cycle is "bad" if it has 4 distinct colors.
    # We can maintain a count of distinct colors per cycle? That's expensive to update.
    # Instead, we just recalculate the status of the affected cycles.
    
    # State:
    # colors: list of 120 ints
    # color_counts: list of 120 ints (count of each color used)
    # num_distinct: int
    # violations: int (count of bad cycles)
    
    # Helper to get distinct colors in a cycle
    def get_cycle_colors(cycle_idx):
        c = _CYCLES[cycle_idx]
        return [colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]]]
    
    def is_bad_cycle(cycle_idx):
        c = _CYCLES[cycle_idx]
        c0, c1, c2, c3 = colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]]
        return len({c0, c1, c2, c3}) == 4

    # Initial violations check (should be 0)
    for ci in range(NUM_CYCLES):
        if is_bad_cycle(ci):
            violations += 1
            
    best_colors = colors[:]
    best_distinct = num_distinct
    
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    max_iter = 50000
    
    # Weights for the cost function
    # Cost = violations * W_V - num_distinct * W_D
    # We want to minimize Cost.
    # W_V must be large enough to prioritize validity.
    # W_D encourages diversity.
    W_V = 100
    W_D = 1
    
    for iteration in range(max_iter):
        if T < T_min:
            break
            
        # Pick a random edge
        ei = random.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Choose new color
        # Strategy: 
        # 1. If we want to increase diversity, pick a color not currently used, or rarely used.
        # 2. If we want to fix violations, pick a color that matches neighbors in bad cycles.
        # Let's mix:
        if random.random() < 0.5:
            # Try to introduce a new color or a less common one
            # Pick a random color from 0..119
            new_color = random.randint(0, 119)
        else:
            # Pick a color already in use (to potentially merge and reduce complexity, or just explore)
            new_color = random.choice(colors)
            
        if new_color == old_color:
            continue
            
        # Calculate delta violations
        delta_violations = 0
        for ci in _EDGE_TO_CYCLES[ei]:
            # Check old state
            c = _CYCLES[ci]
            old_colors_set = set()
            for e in c:
                if e == ei:
                    old_colors_set.add(old_color)
                else:
                    old_colors_set.add(colors[e])
            old_bad = (len(old_colors_set) == 4)
            
            # Check new state
            new_colors_set = set()
            for e in c:
                if e == ei:
                    new_colors_set.add(new_color)
                else:
                    new_colors_set.add(colors[e])
            new_bad = (len(new_colors_set) == 4)
            
            if old_bad and not new_bad:
                delta_violations -= 1
            elif not old_bad and new_bad:
                delta_violations += 1
        
        # Calculate delta distinct
        old_distinct = num_distinct
        # Update counts temporarily to check distinct change
        # If old_color count becomes 0, distinct decreases by 1
        # If new_color count was 0, distinct increases by 1
        delta_distinct = 0
        if color_counts[old_color] == 1:
            delta_distinct -= 1
        if color_counts[new_color] == 0:
            delta_distinct += 1
            
        # Calculate delta cost
        delta_cost = delta_violations * W_V - delta_distinct * W_D
        
        # Acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            # Accept with probability exp(-delta_cost / T)
            # Note: delta_cost is positive here
            prob = math.exp(-delta_cost / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            # Update state
            colors[ei] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            violations += delta_violations
            num_distinct += delta_distinct
            
            # Track best
            # We prefer valid solutions (violations==0) with max distinct.
            # If violations > 0, we don't update best yet, unless we have no valid solution?
            # The task requires "No four-cycle may have four distinct edge colors".
            # So invalid solutions are not acceptable as final output.
            # We should only update best if violations == 0.
            if violations == 0:
                if num_distinct > best_distinct:
                    best_distinct = num_distinct
                    best_colors = colors[:]
        else:
            # Revert (do nothing, as we didn't change state)
            pass
            
        T *= cooling
        
    # If we didn't find a valid solution, best_colors might be invalid.
    # We need to ensure we return a valid solution.
    # If best_distinct is 0 (no valid solution found), we should try a repair phase.
    # However, starting from all 0s guarantees a valid solution.
    # So best_distinct will be at least 1.
    
    # Final check: ensure best_colors is valid
    # (It is, by construction of the update logic)
    
    return best_colors

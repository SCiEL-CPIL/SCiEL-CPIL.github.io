"""New program: hypergraph coloring with entropy-guided simulated annealing."""
import random
import math
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
NUM_COLORS = 120

def _build_edge_index():
    """Map (i,j) with i<j to a linear index 0..119."""
    edge_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx += 1
    return edge_to_idx

def _build_cycles():
    """
    For each 4-vertex subset, compute the 3 cycles.
    Each cycle is a tuple of 4 edge indices.
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
    
    # Initialize with a valid low-diversity coloring
    colors = [0] * NUM_EDGES
    color_counts = [0] * NUM_COLORS
    color_counts[0] = NUM_EDGES
    num_distinct = 1
    violations = 0
    
    # Precompute cycle color counts for fast delta calculation
    # For each cycle, maintain count of each color in that cycle
    # This allows O(1) check of whether a cycle has 4 distinct colors
    
    # cycle_color_counts[ci][color] = count of that color in cycle ci
    cycle_color_counts = [[0] * NUM_COLORS for _ in range(NUM_CYCLES)]
    for ci, cycle in enumerate(_CYCLES):
        for ei in cycle:
            cycle_color_counts[ci][colors[ei]] += 1
    
    # cycle_distinct[ci] = number of distinct colors in cycle ci
    cycle_distinct = [0] * NUM_CYCLES
    for ci in range(NUM_CYCLES):
        for c in range(NUM_COLORS):
            if cycle_color_counts[ci][c] > 0:
                cycle_distinct[ci] += 1
    
    # A cycle is bad if distinct == 4
    violations = sum(1 for d in cycle_distinct if d == 4)
    
    best_colors = colors[:]
    best_distinct = num_distinct
    
    # Entropy-based cost: we want to maximize diversity
    # Use a potential that rewards having many colors with small counts
    # Potential = sum over used colors of f(count) where f is concave
    # f(k) = k * log(k) is convex, so we use -k*log(k) to reward spread
    # But simpler: just use num_distinct as the main driver
    
    T = 5.0
    T_min = 0.001
    cooling = 0.995
    max_iter = 80000
    
    W_V = 1000  # Large weight for violations to ensure validity
    W_D = 1     # Weight for diversity
    
    # Track best
    for iteration in range(max_iter):
        if T < T_min:
            break
            
        # Pick a random edge
        ei = random.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Choose new color with bias toward unused or rare colors
        # 60% chance: pick from unused colors
        # 40% chance: pick from all colors
        if random.random() < 0.6:
            # Pick a color that is currently unused or rarely used
            unused_or_rare = [c for c in range(NUM_COLORS) if color_counts[c] <= 1]
            if unused_or_rare:
                new_color = random.choice(unused_or_rare)
            else:
                new_color = random.randint(0, NUM_COLORS - 1)
        else:
            new_color = random.randint(0, NUM_COLORS - 1)
            
        if new_color == old_color:
            continue
            
        # Calculate delta violations using cycle_color_counts
        delta_violations = 0
        for ci in _EDGE_TO_CYCLES[ei]:
            old_distinct = cycle_distinct[ci]
            
            # Update old_color count
            cycle_color_counts[ci][old_color] -= 1
            if cycle_color_counts[ci][old_color] == 0:
                new_distinct_after_old = old_distinct - 1
            else:
                new_distinct_after_old = old_distinct
            
            # Update new_color count
            old_new_count = cycle_color_counts[ci][new_color]
            cycle_color_counts[ci][new_color] += 1
            if old_new_count == 0:
                new_distinct = new_distinct_after_old + 1
            else:
                new_distinct = new_distinct_after_old
            
            # Revert for next iteration's delta calc (we'll apply if accepted)
            # Actually, we need to be careful: we're computing delta, not applying
            # Let me redo this properly
            
            # Revert the temporary changes
            cycle_color_counts[ci][new_color] -= 1
            cycle_color_counts[ci][old_color] += 1
            
            # Now compute properly
            old_distinct = cycle_distinct[ci]
            
            # After removing old_color:
            if cycle_color_counts[ci][old_color] == 1:
                temp_distinct = old_distinct - 1
            else:
                temp_distinct = old_distinct
            
            # After adding new_color:
            if cycle_color_counts[ci][new_color] == 0:
                new_distinct = temp_distinct + 1
            else:
                new_distinct = temp_distinct
            
            old_bad = (old_distinct == 4)
            new_bad = (new_distinct == 4)
            
            if old_bad and not new_bad:
                delta_violations -= 1
            elif not old_bad and new_bad:
                delta_violations += 1
        
        # Calculate delta distinct
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
            prob = math.exp(-delta_cost / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            # Apply the move
            colors[ei] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            # Update cycle_color_counts and cycle_distinct
            for ci in _EDGE_TO_CYCLES[ei]:
                old_distinct = cycle_distinct[ci]
                
                cycle_color_counts[ci][old_color] -= 1
                if cycle_color_counts[ci][old_color] == 0:
                    temp_distinct = old_distinct - 1
                else:
                    temp_distinct = old_distinct
                
                old_new_count = cycle_color_counts[ci][new_color]
                cycle_color_counts[ci][new_color] += 1
                if old_new_count == 0:
                    new_distinct = temp_distinct + 1
                else:
                    new_distinct = temp_distinct
                
                cycle_distinct[ci] = new_distinct
            
            violations += delta_violations
            num_distinct += delta_distinct
            
            # Track best (only valid solutions)
            if violations == 0:
                if num_distinct > best_distinct:
                    best_distinct = num_distinct
                    best_colors = colors[:]
        
        T *= cooling
        
    return best_colors

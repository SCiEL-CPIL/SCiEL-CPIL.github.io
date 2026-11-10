import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_cycles():
    """Build all 4-cycles and which edges they use."""
    # For each set of 4 vertices, there are 3 different 4-cycles
    # Vertices a,b,c,d: cycles are
    # (a,b,c,d): ab, bc, cd, da
    # (a,b,d,c): ab, bd, dc, ca
    # (a,c,b,d): ac, cb, bd, da
    cycles = []  # list of tuples of 4 edge indices
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
        # Cycle 2: a-b-d-c-a
        e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
        # Cycle 3: a-c-b-d-a
        e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
        
        for cyc in (e1, e2, e3):
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles) - 1)
    
    return cycles, edge_to_cycles

def is_valid_change(colors, edge_idx, new_color, edge_to_cycles):
    """Check if setting edge to new_color keeps all cycles valid."""
    for ci in edge_to_cycles[edge_idx]:
        # Check this cycle
        colors_in_cycle = []
        # We need to check all 4 edges in the cycle
        # But we don't have direct access to cycle edges here, so we check differently
        pass
    # Better: precompute cycle edges
    return True

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin
    
    cycles, edge_to_cycles = build_cycles()
    
    # Precompute: for each edge, store the cycles it belongs to with their full edge lists
    # cycles is already a list of tuples of 4 edge indices
    
    def count_distinct(colors):
        return len(set(colors))
    
    def check_cycle_valid(colors, cyc):
        """A cycle is valid if it does NOT have 4 distinct colors."""
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        return not (len({c0, c1, c2, c3}) == 4)
    
    def is_fully_valid(colors):
        for cyc in cycles:
            if not check_cycle_valid(colors, cyc):
                return False
        return True
    
    def can_set_edge(colors, edge_idx, new_color):
        """Check if setting edge_idx to new_color keeps all affected cycles valid."""
        old_color = colors[edge_idx]
        if old_color == new_color:
            return True
        colors[edge_idx] = new_color
        valid = True
        for ci in edge_to_cycles[edge_idx]:
            if not check_cycle_valid(colors, cycles[ci]):
                valid = False
                break
        colors[edge_idx] = old_color
        return valid
    
    # Greedy: start all zeros, then try to assign new colors
    colors = [0] * NUM_EDGES
    best_colors = colors[:]
    best_count = 1
    max_color_used = 1  # next new color to try
    
    # Greedy phase: try to introduce new colors one by one
    for attempt in range(3000):
        if time.time() - start_time > DEADLINE:
            break
        if max_color_used >= 120:
            break
        new_color = max_color_used
        found = False
        for edge_idx in range(NUM_EDGES):
            if can_set_edge(colors, edge_idx, new_color):
                colors[edge_idx] = new_color
                max_color_used += 1
                found = True
                break
        if not found:
            # Try more aggressively: pick random edge, try all unused colors
            tried = 0
            while tried < 50 and max_color_used < 120:
                tried += 1
                if time.time() - start_time > DEADLINE:
                    break
                edge_idx = random.randint(0, NUM_EDGES - 1)
                new_color = max_color_used
                if can_set_edge(colors, edge_idx, new_color):
                    colors[edge_idx] = new_color
                    max_color_used += 1
                    found = True
                    break
            if not found:
                break
    
    current_count = count_distinct(colors)
    if current_count > best_count:
        best_count = current_count
        best_colors = colors[:]
    
    # Local search: try to improve by swapping colors or introducing new ones
    for restart in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Start from best
        colors = best_colors[:]
        current_count = count_distinct(colors)
        
        for step in range(5000):
            if time.time() - start_time > DEADLINE:
                break
            
            # Random strategy: pick an edge, try a new color
            edge_idx = random.randint(0, NUM_EDGES - 1)
            
            # Try to assign a completely new color
            if max_color_used < 120:
                if can_set_edge(colors, edge_idx, max_color_used):
                    colors[edge_idx] = max_color_used
                    max_color_used += 1
                    new_count = count_distinct(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                    continue
            
            # Try to recolor to an existing color to free up potential
            # Or just try random new colors within range
            new_color = random.randint(0, 119)
            if new_color == colors[edge_idx]:
                continue
            if can_set_edge(colors, edge_idx, new_color):
                colors[edge_idx] = new_color
                new_count = count_distinct(colors)
                if new_count > current_count:
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors[:]
    
    return best_colors

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END

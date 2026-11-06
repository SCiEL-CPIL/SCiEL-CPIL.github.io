import time
import random
import math
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles (three per 4-vertex subset)
CYCLES = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                CYCLES.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                CYCLES.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                CYCLES.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

NUM_CYCLES = len(CYCLES)

# Edge to cycles mapping
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for c_idx, cycle in enumerate(CYCLES):
    for e in cycle:
        edge_to_cycles[e].append(c_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Strategy: Global structural decomposition
    # Phase 1: Find edge-disjoint 4-cycles and assign each a unique color
    # Phase 2: Greedily color remaining edges
    
    current_colors = [0] * NUM_EDGES
    
    # Phase 1: Pack edge-disjoint cycles
    # Use a greedy matching approach on cycles
    # For each cycle, check if all 4 edges are still "free" (unassigned to a previous cycle)
    
    # We'll use a randomized greedy packing: shuffle cycles, try to pack
    # Multiple restarts with different orderings
    
    best_packed_cycles = []
    best_remaining_edges = set(range(NUM_EDGES))
    
    # Try multiple random orderings of cycles
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        cycle_order = list(range(NUM_CYCLES))
        random.shuffle(cycle_order)
        
        packed = []
        used_edges = set()
        
        for c_idx in cycle_order:
            cycle = CYCLES[c_idx]
            # Check if all edges in this cycle are unused
            if all(e not in used_edges for e in cycle):
                packed.append(c_idx)
                for e in cycle:
                    used_edges.add(e)
        
        if len(packed) > len(best_packed_cycles):
            best_packed_cycles = packed
            best_remaining_edges = set(range(NUM_EDGES)) - used_edges
    
    # Assign unique colors to packed cycles
    color_assignment = {}  # edge -> color
    for i, c_idx in enumerate(best_packed_cycles):
        color = i  # color 0, 1, 2, ...
        for e in CYCLES[c_idx]:
            color_assignment[e] = color
    
    num_packed_colors = len(best_packed_cycles)
    
    # Phase 2: Greedily color remaining edges
    # For each remaining edge, try to assign a color such that no cycle gets 4 distinct colors
    
    # Maintain cycle state: for each cycle, track which colors are present
    cycle_colors = [set() for _ in range(NUM_CYCLES)]
    
    # Initialize with packed cycles
    for c_idx in best_packed_cycles:
        col = best_packed_cycles.index(c_idx)
        cycle_colors[c_idx].add(col)
    
    # For cycles that share edges with packed cycles, they already have that color
    # Actually, we need to be more careful. Let's rebuild cycle_colors from scratch.
    
    cycle_colors = [set() for _ in range(NUM_CYCLES)]
    for e, col in color_assignment.items():
        for c_idx in edge_to_cycles[e]:
            cycle_colors[c_idx].add(col)
    
    # Now greedily assign colors to remaining edges
    remaining = list(best_remaining_edges)
    random.shuffle(remaining)
    
    next_color = num_packed_colors
    
    for e in remaining:
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid color for this edge
        # A color c is valid if for every cycle containing e, adding c doesn't make it 4-distinct
        # i.e., for every cycle c_idx containing e: len(cycle_colors[c_idx] | {c}) <= 3
        
        # First, try reusing existing colors
        used_colors = set(color_assignment.values())
        
        found = False
        # Try existing colors first (sorted by how "safe" they are)
        for c in sorted(used_colors):
            valid = True
            for c_idx in edge_to_cycles[e]:
                new_set = cycle_colors[c_idx] | {c}
                if len(new_set) > 3:
                    valid = False
                    break
            if valid:
                color_assignment[e] = c
                for c_idx in edge_to_cycles[e]:
                    cycle_colors[c_idx].add(c)
                found = True
                break
        
        if not found:
            # Try a new color
            if next_color < MAX_COLOR:
                valid = True
                for c_idx in edge_to_cycles[e]:
                    new_set = cycle_colors[c_idx] | {next_color}
                    if len(new_set) > 3:
                        valid = False
                        break
                if valid:
                    color_assignment[e] = next_color
                    next_color += 1
                    for c_idx in edge_to_cycles[e]:
                        cycle_colors[c_idx].add(next_color - 1)
                    found = True
        
        if not found:
            # Can't assign any valid color; leave as 0 (will be checked later)
            # Actually, we need to assign SOMETHING. Assign 0 and hope it works,
            # or try harder.
            color_assignment[e] = 0
            for c_idx in edge_to_cycles[e]:
                cycle_colors[c_idx].add(0)
    
    # Build final color array
    for e in range(NUM_EDGES):
        current_colors[e] = color_assignment.get(e, 0)
    
    # Phase 3: Local improvement - try to increase color count by reassigning edges
    # Simulated annealing style: pick a random edge, try a random new color
    
    def count_distinct_colors():
        return len(set(current_colors))
    
    best_score = count_distinct_colors()
    best_colors = current_colors[:]
    
    # Precompute for speed: for each edge, list of cycles
    # For each cycle, we need to know the 4 edge colors
    
    # Use numpy for vectorized distinct count checks
    # Actually, let's use a simple approach: maintain a count matrix
    
    # For the local search, we'll use a faster validity check
    # Precompute cycle edge indices as numpy arrays
    cycle_edges_np = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)
    
    # Local search
    temp = 1.0
    cooling = 0.9999
    
    while time.time() < deadline - 2:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Try a random new color
        if random.random() < 0.3:
            # Try a completely new color
            new_color = random.randint(0, MAX_COLOR - 1)
        else:
            # Try a random existing or nearby color
            new_color = random.randint(0, MAX_COLOR - 1)
        
        if new_color == old_color:
            continue
        
        # Check validity: for each cycle containing e, check if adding new_color
        # (and removing old_color if no other edge in cycle has it) keeps distinct <= 3
        
        valid = True
        affected_cycles = edge_to_cycles[e]
        
        for c_idx in affected_cycles:
            cycle = CYCLES[c_idx]
            # Get colors of the 4 edges
            cols = [current_colors[ed] for ed in cycle]
            # Replace e's color with new_color
            for i, ed in enumerate(cycle):
                if ed == e:
                    cols[i] = new_color
                    break
            if len(set(cols)) > 3:
                valid = False
                break
        
        if valid:
            current_colors[e] = new_color
            new_score = count_distinct_colors()
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors[:]
        else:
            # Metropolis acceptance (for exploration)
            if random.random() < math.exp(-1.0 / max(temp, 0.01)):
                current_colors[e] = new_color
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best_colors

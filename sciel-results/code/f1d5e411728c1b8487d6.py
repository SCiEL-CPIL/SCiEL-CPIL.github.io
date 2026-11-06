import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute all 4-cycles as tuples of 4 edge indices
    # A 4-cycle on vertices {a,b,c,d} has 3 possible cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d)
    all_4cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        all_4cycles.append((get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)))
        # Cycle 2: a-b-d-c-a
        all_4cycles.append((get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)))
        # Cycle 3: a-c-b-d-a
        all_4cycles.append((get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)))
    
    # For each edge, store the list of 4-cycles it participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for cycle in all_4cycles:
        for e in cycle:
            edge_to_cycles[e].append(cycle)
    
    # Precompute all edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def is_valid(colors):
        """Check if coloring has no rainbow 4-cycle."""
        for cycle in all_4cycles:
            c1, c2, c3, c4 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                return False
        return True
    
    def check_edge_change(colors, edge_idx, new_color):
        """Check if changing edge edge_idx to new_color keeps validity."""
        # Only need to check 4-cycles containing this edge
        for cycle in edge_to_cycles[edge_idx]:
            c1 = colors[cycle[0]] if cycle[0] != edge_idx else new_color
            c2 = colors[cycle[1]] if cycle[1] != edge_idx else new_color
            c3 = colors[cycle[2]] if cycle[2] != edge_idx else new_color
            c4 = colors[cycle[3]] if cycle[3] != edge_idx else new_color
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                return False
        return True
    
    # Start with a simple valid coloring: all edges color 0
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    random.seed(42)
    
    # Local search: try to introduce new colors by changing edge colors
    # Strategy: pick random edge, try random new color, check validity
    
    best_colors = colors[:]
    best_count = 1
    
    # Phase 1: Greedy introduction of new colors
    for attempt in range(100000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random edge
        edge_idx = random.randint(0, NUM_EDGES - 1)
        
        # Try to change to a new unused color
        unused = [c for c in range(120) if c not in used_colors]
        if not unused:
            break
        
        new_color = random.choice(unused)
        
        if check_edge_change(colors, edge_idx, new_color):
            colors[edge_idx] = new_color
            used_colors.add(new_color)
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                best_colors = colors[:]
    
    # Phase 2: Try to swap colors to free up constraints and introduce more
    for attempt in range(200000):
        if time.time() > deadline - 3:
            break
        
        # Pick two random edges
        e1 = random.randint(0, NUM_EDGES - 1)
        e2 = random.randint(0, NUM_EDGES - 1)
        if e1 == e2:
            continue
        
        c1 = colors[e1]
        c2 = colors[e2]
        if c1 == c2:
            continue
        
        # Try swapping
        colors[e1] = c2
        colors[e2] = c1
        
        if is_valid(colors):
            # After swap, try to introduce a new color on one of these edges
            unused = [c for c in range(120) if c not in used_colors]
            if unused:
                for e in [e1, e2]:
                    nc = random.choice(unused)
                    if check_edge_change(colors, e, nc):
                        colors[e] = nc
                        used_colors.add(nc)
                        break
        else:
            # Revert swap
            colors[e1] = c1
            colors[e2] = c2
        
        if len(used_colors) > best_count:
            best_count = len(used_colors)
            best_colors = colors[:]
    
    # Phase 3: Random restarts with different initial patterns
    for restart in range(20):
        if time.time() > deadline - 2:
            break
        
        # Create a random initial coloring with few colors
        num_init_colors = random.randint(2, 6)
        colors = [random.randint(0, num_init_colors - 1) for _ in range(NUM_EDGES)]
        
        if not is_valid(colors):
            # Try to fix by simplifying
            for _ in range(100):
                if is_valid(colors):
                    break
                bad_edge = random.randint(0, NUM_EDGES - 1)
                colors[bad_edge] = random.randint(0, num_init_colors - 1)
        
        if not is_valid(colors):
            continue
        
        used_colors = set(colors)
        local_best = colors[:]
        local_best_count = len(used_colors)
        
        # Local search from this starting point
        for attempt in range(50000):
            if time.time() > deadline - 2:
                break
            
            edge_idx = random.randint(0, NUM_EDGES - 1)
            unused = [c for c in range(120) if c not in used_colors]
            if not unused:
                break
            
            new_color = random.choice(unused)
            
            if check_edge_change(colors, edge_idx, new_color):
                colors[edge_idx] = new_color
                used_colors.add(new_color)
                if len(used_colors) > local_best_count:
                    local_best_count = len(used_colors)
                    local_best = colors[:]
        
        if local_best_count > best_count:
            best_count = local_best_count
            best_colors = local_best[:]
    
    # Final verification
    if not is_valid(best_colors):
        # Fallback to all-zero coloring
        best_colors = [0] * NUM_EDGES
    
    return best_colors
# EVOLVE-BLOCK-END

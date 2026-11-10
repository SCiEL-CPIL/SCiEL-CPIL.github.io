"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct edge coloring of K_16 with no rainbow 4-cycle.
    Maximize number of distinct colors used.
    Uses greedy random assignment with constraint checking.
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute all 4-cycles (all 3 cycles on every 4-vertex subset)
    # A 4-cycle on vertices (a,b,c,d) in order has edges (a,b),(b,c),(c,d),(d,a)
    # But we need all three cycles on every 4-vertex subset:
    # For 4 vertices {a,b,c,d}, the three 4-cycles are:
    #   (a,b,c,d): edges ab,bc,cd,da
    #   (a,b,d,c): edges ab,bd,dc,ca  
    #   (a,c,b,d): edges ac,cb,bd,da
    # Each cycle is defined by 4 edges.
    
    # Build mapping: for each set of 4 vertices, find the 3 cycles
    # Edge index for (i,j) where i<j: idx = i*N - i*(i+1)//2 + (j-i-1)
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i*(i+1)//2 + (j - i - 1)
    
    # Precompute all triples of 4-cycles for each 4-vertex subset
    all_4sets = list(itertools.combinations(range(N), 4))
    # For each 4-set, the three perfect-matchings / 4-cycles:
    # Vertices a,b,c,d. The three ways to form a 4-cycle:
    # Cycle 1: a-b-c-d-a -> edges (a,b),(b,c),(c,d),(d,a)
    # Cycle 2: a-b-d-c-a -> edges (a,b),(b,d),(d,c),(c,a)
    # Cycle 3: a-c-b-d-a -> edges (a,c),(c,b),(b,d),(d,a)
    
    def get_cycles_for_4set(a, b, c, d):
        """Return list of 3 cycles, each cycle is a tuple of 4 edge indices."""
        # Cycle 1: a-b-c-d-a
        e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        # Cycle 2: a-b-d-c-a
        e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
        # Cycle 3: a-c-b-d-a
        e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
        return [e1, e2, e3]
    
    # Precompute all constraints: list of (cycle_edges_tuple, four_vertex_set)
    constraints = []
    for four_set in all_4sets:
        a, b, c, d = four_set
        cycles = get_cycles_for_4set(a, b, c, d)
        for cyc in cycles:
            constraints.append(cyc)
    
    num_edges = N*(N-1)//2  # 120
    
    def is_rainbow(colors, cycle_edges):
        """Check if a 4-cycle has 4 distinct colors."""
        c = [colors[e] for e in cycle_edges]
        return len(set(c)) == 4
    
    def check_constraint(colors, cycle_edges):
        """Check if assigning would create a rainbow cycle."""
        return is_rainbow(colors, cycle_edges)
    
    # Greedy random approach: try multiple random seeds, pick best
    best_result = None
    best_color_count = 0
    
    # Try several random assignments
    for trial in range(50):
        if time.time() > deadline:
            break
        
        random.seed(trial * 12345 + 67890)
        colors = [0] * num_edges
        used_colors = {0}
        
        # Try to assign new colors greedily
        # Process edges in random order
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > deadline:
                break
            
            # Try to assign a new color
            assigned = False
            # Try existing colors first in random order, then new colors
            existing = list(used_colors)
            random.shuffle(existing)
            
            # Try each existing color
            for col in existing:
                colors[e] = col
                # Check all constraints involving edge e
                valid = True
                for cyc in constraints:
                    if e in cyc:
                        if check_constraint(colors, cyc):
                            valid = False
                            break
                if valid:
                    assigned = True
                    break
                colors[e] = 0  # reset
            
            if not assigned:
                # Try new colors
                for col in range(120):
                    if col in used_colors:
                        continue
                    colors[e] = col
                    valid = True
                    for cyc in constraints:
                        if e in cyc:
                            if check_constraint(colors, cyc):
                                valid = False
                                break
                    if valid:
                        used_colors.add(col)
                        assigned = True
                        break
                    colors[e] = 0
                
                if not assigned:
                    # Just use color 0
                    colors[e] = 0
            
            # Count colors
            current_count = len(set(colors))
        
        current_count = len(set(colors))
        if current_count > best_color_count:
            best_color_count = current_count
            best_result = colors[:]
    
    if best_result is None:
        return [0] * num_edges
    return best_result
# EVOLVE-BLOCK-END

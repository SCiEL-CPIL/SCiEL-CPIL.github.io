import time
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Precompute edge index: for i < j, index = i*(N-1) - i*(i-1)//2 + (j - i - 1)
    # Verify: i=0,j=1: 0 + 0 = 0 ✓
    # i=0,j=15: 0 + 14 = 14 ✓ (row 0 has 15 edges: j=1..15)
    # i=1,j=2: 1*15 - 0 + 0 = 15 ✓ (row 1 starts at 15)
    # Last: i=14,j=15: 14*15 - 14*13//2 + 0 = 210 - 91 = 119 ✓
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute for each edge, the list of 4-cycles it belongs to
    # A 4-cycle involves 4 vertices and 4 edges forming a cycle
    # For edge (u,v), a 4-cycle containing it needs two more vertices w,x
    # such that u-w-v-x-u or similar forms a cycle
    # Actually, a 4-cycle on vertices {a,b,c,d} with a<b<c<d has three possible cycles:
    #   (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # Edge (i,j) appears in 4-cycles where i,j are two of the four vertices
    
    # For each edge, find all 4-cycles containing it
    # A 4-cycle is defined by 4 vertices and which 4 of the 6 edges form the cycle
    # For 4 vertices a<b<c<d, the three 4-cycles use edges:
    #   Cycle 1: (a,b),(b,c),(c,d),(d,a) -> edges (a,b),(b,c),(c,d),(a,d)
    #   Cycle 2: (a,b),(b,d),(d,c),(c,a) -> edges (a,b),(b,d),(c,d),(a,c)
    #   Cycle 3: (a,c),(c,b),(b,d),(d,a) -> edges (a,c),(b,c),(b,d),(a,d)
    
    # Precompute: for each edge, list of (cycle_id, other_edges_in_cycle)
    # Better: for each edge, list of sets of 3 other edge indices that form a 4-cycle with it
    
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    # Generate all 4-vertex subsets
    from itertools import combinations
    
    four_cycles = []  # list of (4 vertices, list of 4 edge indices)
    
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        # Three 4-cycles:
        # Cycle 1: a-b-c-d-a
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)]
        # Cycle 2: a-b-d-c-a
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        # Cycle 3: a-c-b-d-a
        e3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]
        
        for cycle_edges in [e1, e2, e3]:
            four_cycles.append(cycle_edges)
            for e in cycle_edges:
                others = [x for x in cycle_edges if x != e]
                edge_to_cycles[e].append(others)
    
    # Initialize with transitive coloring: color of (i,j) = i
    colors = [0] * NUM_EDGES
    for i in range(N - 1):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    # Count distinct colors
    def count_distinct(colors):
        return len(set(colors))
    
    best_colors = colors[:]
    best_count = count_distinct(best_colors)
    
    # Local search: try to change an edge's color to an unused color
    # While maintaining validity
    
    # Precompute which colors are "available" - we can use 0..119
    MAX_COLOR = 120
    
    # For speed, track which colors are used
    def try_change(colors, edge, new_color):
        """Check if changing edge to new_color maintains validity."""
        old_color = colors[edge]
        if old_color == new_color:
            return True
        
        # Check all 4-cycles containing this edge
        for others in edge_to_cycles[edge]:
            cycle_colors = set()
            cycle_colors.add(new_color)
            for e in others:
                cycle_colors.add(colors[e])
            if len(cycle_colors) == 4:
                return False
        return True
    
    # Local search with random restarts and simulated annealing-like approach
    random.seed(42)
    
    current = best_colors[:]
    current_count = best_count
    
    # Try to improve by changing edges to new colors
    # Strategy: for each edge, try assigning it a color that is currently unused
    # to increase the number of distinct colors
    
    while time.time() < deadline:
        # Find unused colors
        used_colors = set(current)
        unused_colors = [c for c in range(MAX_COLOR) if c not in used_colors]
        
        if not unused_colors:
            break
        
        # Shuffle edges for randomization
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        improved = False
        for edge in edges:
            if time.time() > deadline - 5:
                break
            
            old_color = current[edge]
            
            # Try each unused color
            random.shuffle(unused_colors)
            for new_color in unused_colors:
                if try_change(current, edge, new_color):
                    current[edge] = new_color
                    new_count = current_count + 1  # Added a new color
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = current[:]
                    current_count = new_count
                    improved = True
                    break  # Move to next edge
        
        if not improved:
            # Try some random changes to escape local optima
            # Change an edge to any valid color (even if already used)
            any_improved = False
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                edge = random.randint(0, NUM_EDGES - 1)
                new_color = random.randint(0, MAX_COLOR - 1)
                if try_change(current, edge, new_color):
                    old = current[edge]
                    current[edge] = new_color
                    # Recount
                    new_count = len(set(current))
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = current[:]
                    current_count = new_count
                    any_improved = True
                    break
            if not any_improved:
                # Stuck, do a small random perturbation
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    edge = random.randint(0, NUM_EDGES - 1)
                    new_color = random.randint(0, MAX_COLOR - 1)
                    if try_change(current, edge, new_color):
                        current[edge] = new_color
                        current_count = len(set(current))
                        if current_count > best_count:
                            best_count = current_count
                            best_colors = current[:]
                        break
    
    # Verify best_colors is valid
    def is_valid(colors):
        for cycle in four_cycles:
            if len(set(colors[e] for e in cycle)) == 4:
                return False
        return True
    
    if not is_valid(best_colors):
        # Fallback to transitive
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors

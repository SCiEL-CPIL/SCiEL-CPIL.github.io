import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute edge index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return (N - 1) * i - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute all 4-vertex subsets and their three 4-cycles
    # For each 4-set {a,b,c,d} with a<b<c<d, the three cycles are:
    # 1: (a,b,c,d) -> edges (a,b),(b,c),(c,d),(d,a)
    # 2: (a,b,d,c) -> edges (a,b),(b,d),(d,c),(c,a)
    # 3: (a,c,b,d) -> edges (a,c),(c,b),(b,d),(d,a)
    
    four_sets = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three cycles, each is a tuple of 4 edge indices
                    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
                    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
                    e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
                    four_sets.append((e1, e2, e3))
    
    # Precompute for each edge, which 4-cycles it participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3) in enumerate(four_sets):
        for e in e1:
            edge_to_cycles[e].append(ci)
        for e in e2:
            edge_to_cycles[e].append(ci)
        for e in e3:
            edge_to_cycles[e].append(ci)
    
    def is_valid(colors):
        """Check if coloring is valid (no 4-cycle has 4 distinct colors)."""
        for (e1, e2, e3) in four_sets:
            c1 = colors[e1[0]]
            c2 = colors[e1[1]]
            c3 = colors[e1[2]]
            c4 = colors[e1[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
            c1 = colors[e2[0]]
            c2 = colors[e2[1]]
            c3 = colors[e2[2]]
            c4 = colors[e2[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
            c1 = colors[e3[0]]
            c2 = colors[e3[1]]
            c3 = colors[e3[2]]
            c4 = colors[e3[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True
    
    def check_cycle_valid(colors, e_tuple):
        """Check if a specific 4-cycle is valid."""
        c1 = colors[e_tuple[0]]
        c2 = colors[e_tuple[1]]
        c3 = colors[e_tuple[2]]
        c4 = colors[e_tuple[3]]
        return len({c1, c2, c3, c4}) != 4
    
    # Base coloring: color edge (i,j) by i
    colors = [0] * NUM_EDGES
    for i in range(N - 1):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    # Verify base is valid
    assert is_valid(colors), "Base coloring should be valid"
    
    best_colors = colors[:]
    best_num_colors = len(set(best_colors))
    
    # Local search: try to change edges to new or different colors
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try changing to a new color (15 + something) or another existing color
        # First, try to use a new color
        used_colors = set(colors)
        # Try colors from 15 upward (new colors) and also existing ones
        candidates = list(range(120))
        random.shuffle(candidates)
        
        found = False
        for new_color in candidates:
            if new_color == old_color:
                continue
            # Check if this change preserves validity
            colors[e] = new_color
            valid = True
            for ci in edge_to_cycles[e]:
                e1, e2, e3 = four_sets[ci]
                if not check_cycle_valid(colors, e1):
                    valid = False
                    break
                if not check_cycle_valid(colors, e2):
                    valid = False
                    break
                if not check_cycle_valid(colors, e3):
                    valid = False
                    break
            if valid:
                found = True
                break
            else:
                colors[e] = old_color
        
        if found:
            num_colors = len(set(colors))
            if num_colors > best_num_colors:
                best_colors = colors[:]
                best_num_colors = num_colors
        else:
            colors[e] = old_color
        
        # Check time
        if time.time() > deadline - 2:
            break
    
    # Ensure best_colors is valid
    if not is_valid(best_colors):
        # Fallback to base
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors

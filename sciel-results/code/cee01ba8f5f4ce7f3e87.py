"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    Uses local search to maximize distinct colors while
    maintaining the C4-rainbow-free property.
    """
    deadline = time.time() + 50  # Safety margin for 90s limit
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        # Correct lexicographic index formula
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def get_edge(i, j):
        if i > j:
            i, j = j, i
        return edge_idx(i, j)
    
    # Precompute all 4-cycles and their edge indices
    four_cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    c1 = (get_edge(a,b), get_edge(b,c), get_edge(c,d), get_edge(d,a))
                    # Cycle 2: a-b-d-c-a
                    c2 = (get_edge(a,b), get_edge(b,d), get_edge(d,c), get_edge(c,a))
                    # Cycle 3: a-c-b-d-a
                    c3 = (get_edge(a,c), get_edge(c,b), get_edge(b,d), get_edge(d,a))
                    four_cycles.append((c1, c2, c3))
    
    # Initialize with a valid coloring: all 0s
    colors = [0] * 120
    
    def is_valid(col):
        for c1, c2, c3 in four_cycles:
            if len({col[c1[0]], col[c1[1]], col[c1[2]], col[c1[3]]}) == 4:
                return False
            if len({col[c2[0]], col[c2[1]], col[c2[2]], col[c2[3]]}) == 4:
                return False
            if len({col[c3[0]], col[c3[1]], col[c3[2]], col[c3[3]]}) == 4:
                return False
        return True
    
    def count_distinct(col):
        return len(set(col))
    
    best_colors = colors[:]
    best_count = count_distinct(best_colors)
    
    random.seed(42)
    edge_list = list(range(120))
    
    # Greedy addition of colors
    # Try to assign new colors to edges if it maintains validity
    # Start with 1 color, try to add more
    
    # To speed up is_valid, we can check only affected cycles, but full check is fine for 120 edges
    # 1820 cycles * 3 checks * 4 edges = ~22k operations. Very fast.
    
    while time.time() < deadline:
        random.shuffle(edge_list)
        improved = False
        
        for ei in edge_list:
            if time.time() > deadline:
                break
            
            old_color = colors[ei]
            found_improvement = False
            
            # Try all possible colors
            # Optimization: Only try colors that are currently used + 1 new color
            # But we want to maximize distinct count, so we should try to introduce new colors
            # or swap colors to free up space?
            # Simple hill climbing: try any color.
            
            for new_color in range(120):
                if new_color == old_color:
                    continue
                
                colors[ei] = new_color
                
                if is_valid(colors):
                    new_count = count_distinct(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                        found_improvement = True
                        improved = True
                        break
                else:
                    colors[ei] = old_color
            
            if not found_improvement:
                colors[ei] = old_color
        
        if not improved:
            # If no improvement in a full pass, try random restarts or break
            # Given the budget, breaking is safer than infinite loop
            break
    
    return best_colors
# EVOLVE-BLOCK-END

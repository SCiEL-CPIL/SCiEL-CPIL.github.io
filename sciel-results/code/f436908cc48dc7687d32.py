"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    Uses structured initialization and targeted local search to maximize
    distinct colors while maintaining the C4-rainbow-free property.
    """
    deadline = time.time() + 55  # Safety margin for 90s limit
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def get_edge(i, j):
        if i > j:
            i, j = j, i
        return edge_idx(i, j)
    
    # Precompute all 4-cycles
    all_cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    all_cycles.append([get_edge(a,b), get_edge(b,c), get_edge(c,d), get_edge(d,a)])
                    all_cycles.append([get_edge(a,b), get_edge(b,d), get_edge(d,c), get_edge(c,a)])
                    all_cycles.append([get_edge(a,c), get_edge(c,b), get_edge(b,d), get_edge(d,a)])
    
    cycles_arr = np.array(all_cycles, dtype=np.int32)
    num_cycles = len(cycles_arr)
    
    # Precompute for each edge, which cycles contain it
    edge_to_cycles = [[] for _ in range(120)]
    for ci, cycle in enumerate(cycles_arr):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    
    def is_valid_full(colors_np):
        """Vectorized full validity check."""
        cycle_colors = colors_np[cycles_arr]
        sorted_colors = np.sort(cycle_colors, axis=1)
        diffs = np.abs(np.diff(sorted_colors, axis=1))
        rainbow_mask = np.all(diffs > 0, axis=1)
        return not np.any(rainbow_mask)
    
    def is_valid_incremental(colors_np, changed_edge):
        """Check only cycles containing the changed edge."""
        cycle_indices = edge_to_cycles[changed_edge]
        if not cycle_indices:
            return True
        cycle_colors = colors_np[cycles_arr[cycle_indices]]
        sorted_colors = np.sort(cycle_colors, axis=1)
        diffs = np.abs(np.diff(sorted_colors, axis=1))
        rainbow_mask = np.all(diffs > 0, axis=1)
        return not np.any(rainbow_mask)
    
    def count_distinct(colors_list):
        return len(set(colors_list))
    
    # Structured initial colorings
    def initial_coloring_min(i, j):
        """Color by min vertex index."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                col[get_edge(i, j)] = min(i, j)
        return col
    
    def initial_coloring_sum_mod(i, j):
        """Color by (i+j) mod k."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                col[get_edge(i, j)] = (i + j) % 10
        return col
    
    def initial_coloring_diff_mod(i, j):
        """Color by (j-i) mod k."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                col[get_edge(i, j)] = (j - i) % 12
        return col
    
    def initial_coloring_product_mod(i, j):
        """Color by (i*j) mod k."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                col[get_edge(i, j)] = (i * j) % 15
        return col
    
    def initial_coloring_bipartite(i, j):
        """Color based on bipartition of vertices."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                # Split into two groups
                if i < 8 and j < 8:
                    col[get_edge(i, j)] = 0
                elif i >= 8 and j >= 8:
                    col[get_edge(i, j)] = 1
                else:
                    col[get_edge(i, j)] = 2 + (i % 4)
        return col
    
    # Try multiple initial colorings
    initializers = [
        initial_coloring_min,
        initial_coloring_sum_mod,
        initial_coloring_diff_mod,
        initial_coloring_product_mod,
        initial_coloring_bipartite,
    ]
    
    best_colors = None
    best_count = 0
    random.seed(42)
    np.random.seed(42)
    
    for init_func in initializers:
        if time.time() > deadline - 30:
            break
        
        colors = init_func(0, 0)
        colors_np = np.array(colors, dtype=np.int32)
        
        # Verify initial coloring
        if not is_valid_full(colors_np):
            continue
        
        cnt = count_distinct(colors)
        if cnt > best_count:
            best_count = cnt
            best_colors = colors[:]
        
        # Local search: try to add new colors
        # Strategy 1: Try each edge with new colors
        for ei in range(120):
            if time.time() > deadline - 25:
                break
            old_color = colors[ei]
            # Try unused colors first
            used_colors = set(colors)
            for new_color in range(119, -1, -1):
                if new_color in used_colors:
                    continue
                colors[ei] = new_color
                colors_np[ei] = new_color
                if is_valid_incremental(colors_np, ei):
                    new_count = count_distinct(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                    break
                colors[ei] = old_color
                colors_np[ei] = old_color
        
        # Strategy 2: Recolor edges to create opportunities for new colors
        # Pick an edge that has a unique color, try to recolor it to match another
        for _ in range(50):  # Limited iterations
            if time.time() > deadline - 20:
                break
            
            # Find edges with rare colors
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            # Pick a color that appears only once
            rare_colors = [c for c, cnt in color_counts.items() if cnt == 1]
            if not rare_colors:
                break
            
            rare_color = random.choice(rare_colors)
            # Find the edge with this color
            target_edge = None
            for ei in range(120):
                if colors[ei] == rare_color:
                    target_edge = ei
                    break
            
            if target_edge is None:
                break
            
            # Try to recolor this edge to match a neighboring edge's color
            # This might free up the rare color for a new edge
            for new_color in set(colors):
                if new_color == rare_color:
                    continue
                old_color = colors[target_edge]
                colors[target_edge] = new_color
                colors_np[target_edge] = new_color
                if is_valid_incremental(colors_np, target_edge):
                    # Now try to use the freed rare color on another edge
                    for ei2 in range(120):
                        if ei2 == target_edge:
                            continue
                        if colors[ei2] == rare_color:
                            continue
                        old2 = colors[ei2]
                        colors[ei2] = rare_color
                        colors_np[ei2] = rare_color
                        if is_valid_incremental(colors_np, ei2):
                            new_count = count_distinct(colors)
                            if new_count > best_count:
                                best_count = new_count
                                best_colors = colors[:]
                            break
                        colors[ei2] = old2
                        colors_np[ei2] = old2
                    break
                colors[target_edge] = old_color
                colors_np[target_edge] = old_color
    
    # Final polish: try all edges with all possible new colors
    if best_colors is not None:
        colors = best_colors[:]
        colors_np = np.array(colors, dtype=np.int32)
        
        for ei in range(120):
            if time.time() > deadline - 5:
                break
            old_color = colors[ei]
            used_colors = set(colors)
            for new_color in range(119, -1, -1):
                if new_color in used_colors:
                    continue
                colors[ei] = new_color
                colors_np[ei] = new_color
                if is_valid_incremental(colors_np, ei):
                    new_count = count_distinct(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                    break
                colors[ei] = old_color
                colors_np[ei] = old_color
    
    if best_colors is None:
        # Fallback: all zeros is always valid
        best_colors = [0] * 120
    
    return best_colors
# EVOLVE-BLOCK-END

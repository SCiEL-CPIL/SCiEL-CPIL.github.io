import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 45  # seconds, safety margin before 60s kill

def construct():
    start_time = time.time()
    
    # Build edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edge_list.append((i, j))
            idx += 1
    
    # Enumerate all 4-sets
    all_4sets = list(itertools.combinations(range(N), 4))
    
    # For each 4-set, precompute the three cycles as tuples of edge indices
    def get_cycles(a, b, c, d):
        e1 = (edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)])
        e2 = (edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)])
        e3 = (edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)])
        return (e1, e2, e3)
    
    # For each edge, store the list of (4set_index, cycle_index) pairs that involve it
    edge_to_4sets = [[] for _ in range(NUM_EDGES)]
    cycles_by_4set = []
    for s_idx, (a, b, c, d) in enumerate(all_4sets):
        cycles = get_cycles(a, b, c, d)
        cycles_by_4set.append(cycles)
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_4sets[e_idx].append((s_idx, c_idx))
    
    def is_valid_cycle(colors, cycle):
        """Check if a cycle (tuple of 4 edge indices) has < 4 distinct colors"""
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
        return True
    
    def check_edge_validity(colors, e_idx):
        """Check all 4-sets/cycles involving edge e_idx"""
        for (s_idx, c_idx) in edge_to_4sets[e_idx]:
            cycle = cycles_by_4set[s_idx][c_idx]
            if not is_valid_cycle(colors, cycle):
                return False
        return True
    
    def full_validation(colors):
        """Full validation of all cycles"""
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    return False
        return True
    
    # Initialize with the known valid coloring: color(i,j) = i for i<j
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edge_list):
        colors[idx] = i
    
    # Verify initial coloring is valid
    if not full_validation(colors):
        return [0] * NUM_EDGES
    
    # Systematic local search: try recoloring edges to new colors
    # Start with the base 15 colors (0-14), try to add more
    current_colors = set(colors)
    num_colors = len(current_colors)
    
    # Precompute which edges can potentially be recolored
    # An edge (i,j) with i<j currently has color i
    # Try recoloring to new colors 15, 16, 17, ...
    
    best_colors = colors[:]
    best_num_colors = num_colors
    
    # Strategy: iteratively try to add new colors by recoloring individual edges
    # For each candidate new color, try each edge and see if recoloring is valid
    
    max_new_color = 119
    new_color_pool = list(range(15, max_new_color + 1))
    
    # Try to add colors one at a time
    added_colors = []
    for new_color in new_color_pool:
        if time.time() - start_time > DEADLINE:
            break
        
        found = False
        # Try each edge in order (prioritize edges that appear in fewer constraints)
        # Sort edges by number of constraints (fewer = easier to recolor)
        edge_constraint_counts = [(len(edge_to_4sets[e]), e) for e in range(NUM_EDGES)]
        edge_constraint_counts.sort()
        
        for _, e_idx in edge_constraint_counts:
            if time.time() - start_time > DEADLINE:
                break
            
            old_color = colors[e_idx]
            colors[e_idx] = new_color
            
            if check_edge_validity(colors, e_idx):
                found = True
                added_colors.append((e_idx, old_color, new_color))
                break
            
            colors[e_idx] = old_color
        
        if not found:
            # Try with random edge ordering for diversity
            random.seed(hash(new_color) % 100000)
            shuffled_edges = list(range(NUM_EDGES))
            random.shuffle(shuffled_edges)
            
            for e_idx in shuffled_edges:
                if time.time() - start_time > DEADLINE:
                    break
                
                old_color = colors[e_idx]
                colors[e_idx] = new_color
                
                if check_edge_validity(colors, e_idx):
                    found = True
                    added_colors.append((e_idx, old_color, new_color))
                    break
                
                colors[e_idx] = old_color
    
    # Also try to recolor edges to existing colors that aren't i (to potentially free up constraints)
    # This is a secondary optimization - skip for now to keep it simple
    
    # Final validation
    if not full_validation(colors):
        # Fallback to best known valid
        colors = best_colors
    
    # Ensure all colors are in range 0..119
    for i in range(NUM_EDGES):
        if colors[i] < 0 or colors[i] > 119:
            colors[i] = 0
    
    return colors

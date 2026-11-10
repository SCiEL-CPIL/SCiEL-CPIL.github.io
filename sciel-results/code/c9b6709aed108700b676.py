"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    n = 16
    deadline = time.time() + 140.0  # 140s safety margin
    
    # Initial coloring: (i+j) mod 17
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    # Precompute all K4s
    all_k4s = list(itertools.combinations(range(n), 4))
    
    # Build edge index map: (i,j) -> index in colors list
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx += 1
    
    # Precompute for each edge, the list of K4s containing it
    edge_k4s = [[] for _ in range(len(colors))]
    for k4 in all_k4s:
        for a, b in itertools.combinations(k4, 2):
            idx = edge_to_idx[(a, b)]
            edge_k4s[idx].append(k4)
    
    def count_k4_colors(k4, colors):
        seen = set()
        for a, b in itertools.combinations(k4, 2):
            idx = edge_to_idx[(a, b)]
            seen.add(colors[idx])
        return len(seen)
    
    def is_valid(colors):
        for k4 in all_k4s:
            if count_k4_colors(k4, colors) < 5:
                return False
        return True
    
    # Check initial validity
    if not is_valid(colors):
        return colors
    
    # Local search: try to reduce number of distinct colors
    used_colors = set(colors)
    
    # Try to eliminate colors by recoloring edges
    max_iterations = 50000
    iteration = 0
    random.seed(42)
    
    while iteration < max_iterations and time.time() < deadline:
        iteration += 1
        current_colors = set(colors)
        
        if len(current_colors) < best_num_colors:
            best_colors = colors[:]
            best_num_colors = len(current_colors)
            used_colors = set(colors)
        
        # Pick a random color to try to eliminate
        if len(used_colors) <= 13:
            break
        
        target_color = random.choice(list(used_colors))
        
        # Find edges with this color
        edges_with_color = [idx for idx, c in enumerate(colors) if c == target_color]
        
        if not edges_with_color:
            used_colors.discard(target_color)
            continue
        
        # Try to recolor each edge with this color to another existing color
        for idx in edges_with_color:
            i, j = next(((a, b) for (a, b), val in edge_to_idx.items() if val == idx), None)
            if i is None:
                continue
            
            # Get K4s containing this edge
            affected_k4s = edge_k4s[idx]
            
            # Find colors that would keep all affected K4s valid
            valid_new_colors = []
            for new_color in used_colors:
                if new_color == target_color:
                    continue
                
                # Check if changing this edge to new_color keeps all affected K4s valid
                valid = True
                for k4 in affected_k4s:
                    # Count colors in this K4 with the hypothetical change
                    seen = set()
                    for a, b in itertools.combinations(k4, 2):
                        eidx = edge_to_idx[(a, b)]
                        if eidx == idx:
                            seen.add(new_color)
                        else:
                            seen.add(colors[eidx])
                    if len(seen) < 5:
                        valid = False
                        break
                
                if valid:
                    valid_new_colors.append(new_color)
            
            if valid_new_colors:
                # Pick a random valid new color
                new_color = random.choice(valid_new_colors)
                colors[idx] = new_color
                used_colors.discard(target_color) if all(colors[e] != target_color for e in edges_with_color) else None
                break
        else:
            # No edge could be recolored; try next color
            continue
    
    # Return the best coloring found
    return best_colors
# EVOLVE-BLOCK-END

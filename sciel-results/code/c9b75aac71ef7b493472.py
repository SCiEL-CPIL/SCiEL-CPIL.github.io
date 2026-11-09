import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    n = 16
    # Edge indices
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)  # 120
    
    # Edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # Precompute all K4s
    all_k4s = list(combinations(range(n), 4))
    # For each K4, store the 6 edge indices
    k4_edges = []
    for k4 in all_k4s:
        e_list = []
        for a, b in combinations(k4, 2):
            e_list.append(edge_idx[(a, b)])
        k4_edges.append(e_list)
    
    # Start with seed coloring
    colors = np.array([(i + j) % 17 for i, j in edges], dtype=np.int32)
    
    def count_colors():
        return len(set(colors.tolist()))
    
    def count_violations():
        """Count K4s with fewer than 5 distinct colors."""
        violations = 0
        for e_list in k4_edges:
            c = colors[e_list]
            if len(set(c.tolist())) < 5:
                violations += 1
        return violations
    
    def is_valid():
        return count_violations() == 0
    
    # Initial check
    init_colors = count_colors()
    init_violations = count_violations()
    
    if init_violations > 0:
        # Seed should be valid, but just in case
        pass
    
    best_colors = colors.copy()
    best_num_colors = init_colors
    best_violations = init_violations
    
    # Local search: try to recolor edges to reduce total colors
    # Strategy: pick a color that appears least, try to reassign those edges to other colors
    # that don't create violations
    
    random.seed(42)
    
    for iteration in range(1000000):
        if time.time() > deadline:
            break
        
        # Try to reduce colors
        # Get current color counts
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Sort colors by frequency (ascending)
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
        
        # Try to eliminate the least frequent color
        for target_color, count in sorted_colors[:3]:
            if count > 1:
                continue  # Only try to eliminate singleton or very small colors
            
            # Find edges with this color
            target_edges = [idx for idx, c in enumerate(colors) if c == target_color]
            
            if not target_edges:
                continue
            
            # Try to reassign each target edge to a different color
            success = True
            for e_idx in target_edges:
                # Try all other colors
                found = False
                for new_color in range(0, 20):
                    if new_color == target_color:
                        continue
                    # Check if reassigning creates violations
                    old_c = colors[e_idx]
                    colors[e_idx] = new_color
                    
                    # Check K4s containing this edge
                    # Find which K4s contain edge e_idx
                    # This is expensive, let's just check all K4s for now
                    # Actually, let's be smarter: only check K4s that contain this edge
                    # An edge (i,j) is in K4s {i,j,a,b} for a,b not in {i,j}
                    i, j = edges[e_idx]
                    other_vertices = [v for v in range(n) if v != i and v != j]
                    local_violations = 0
                    for a_idx in range(len(other_vertices)):
                        for b_idx in range(a_idx+1, len(other_vertices)):
                            a = other_vertices[a_idx]
                            b = other_vertices[b_idx]
                            # K4 = {i, j, a, b}
                            # Get the 6 edges
                            k4 = [i, j, a, b]
                            e_list = []
                            for x, y in combinations(k4, 2):
                                if x > y:
                                    x, y = y, x
                                e_list.append(edge_idx[(x, y)])
                            c_vals = colors[e_list]
                            if len(set(c_vals.tolist())) < 5:
                                local_violations += 1
                                break
                        if local_violations > 0:
                            break
                    
                    if local_violations == 0:
                        found = True
                        break
                    else:
                        colors[e_idx] = old_c
                
                if not found:
                    colors[e_idx] = old_c
                    success = False
                    break
            
            if success:
                # Check overall validity
                if count_violations() == 0:
                    new_colors = count_colors()
                    if new_colors < best_num_colors:
                        best_colors = colors.copy()
                        best_num_colors = new_colors
                        best_violations = 0
                    # Continue trying to reduce more
                else:
                    # Revert
                    for e_idx in target_edges:
                        colors[e_idx] = target_color
                break
    
    # Return best result
    if best_violations == 0:
        return best_colors.tolist()
    else:
        # Fallback to seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

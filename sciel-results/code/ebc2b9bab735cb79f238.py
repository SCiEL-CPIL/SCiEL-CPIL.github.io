"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n = 16
    # Build edge index mapping
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute all K4s and their edge indices
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_indices = [edge_to_idx[(a,b)], edge_to_idx[(a,c)], edge_to_idx[(a,d)],
                      edge_to_idx[(b,c)], edge_to_idx[(b,d)], edge_to_idx[(c,d)]]
        k4s.append(e_indices)
    num_k4s = len(k4s)  # 1820
    
    # Initial coloring: (i+j) % 17
    coloring = np.array([(i + j) % 17 for i, j in edges], dtype=np.int32)
    
    # For each color, track which edges use it
    # For each K4, track the set of colors present
    
    def count_colors_used(coloring):
        return len(set(coloring.tolist()))
    
    best_coloring = coloring.copy()
    best_num_colors = count_colors_used(best_coloring)
    
    # Helper: check if a K4 has >= 5 distinct colors
    def k4_ok(k4_edge_indices, coloring):
        colors = set(coloring[k4_edge_indices].tolist())
        return len(colors) >= 5
    
    # For efficiency, precompute for each edge which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4s):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Try to eliminate colors from highest to lowest
    # For each color c, try to reassign edges with color c to other colors
    # such that all K4s remain valid
    
    def try_eliminate_color(coloring, target_color):
        """Try to eliminate target_color by reassigning its edges."""
        edges_with_color = np.where(coloring == target_color)[0]
        if len(edges_with_color) == 0:
            return True, coloring
        
        # Try to reassign each edge to a different color
        # Use greedy: for each edge, pick the color that causes fewest violations
        new_coloring = coloring.copy()
        
        for e_idx in edges_with_color:
            # Find the best new color
            best_new_color = -1
            best_violations = float('inf')
            
            for new_c in range(16):  # Try colors 0..15
                if new_c == target_color:
                    continue
                # Check if changing this edge to new_c violates any K4
                # that was previously OK
                violations = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    # Check current state of this K4
                    e_indices = k4s[k4_idx]
                    # Get colors in this K4 with the proposed change
                    colors = []
                    for ei in e_indices:
                        if ei == e_idx:
                            colors.append(new_c)
                        else:
                            colors.append(new_coloring[ei])
                    if len(set(colors)) < 5:
                        violations += 1
                
                if violations < best_violations:
                    best_violations = violations
                    best_new_color = new_c
                    if violations == 0:
                        break
            
            if best_new_color == -1 or best_violations > 0:
                # Can't eliminate this color
                return False, coloring
            new_coloring[e_idx] = best_new_color
        
        return True, new_coloring
    
    # Iteratively try to eliminate colors
    current_coloring = coloring.copy()
    current_num_colors = count_colors_used(current_coloring)
    
    # Sort colors by frequency (try to eliminate rare colors first)
    while time.time() < deadline:
        color_counts = {}
        for c in current_coloring:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        if not color_counts:
            break
        
        # Try to eliminate the least frequent color
        target_color = min(color_counts.keys(), key=lambda c: color_counts[c])
        
        success, new_coloring = try_eliminate_color(current_coloring, target_color)
        if success:
            current_coloring = new_coloring
            current_num_colors = count_colors_used(current_coloring)
            if current_num_colors < best_num_colors:
                best_num_colors = current_num_colors
                best_coloring = current_coloring.copy()
        else:
            # Try next most frequent
            sorted_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])
            found = False
            for tc in sorted_colors:
                if tc == target_color:
                    continue
                success, new_coloring = try_eliminate_color(current_coloring, tc)
                if success:
                    current_coloring = new_coloring
                    current_num_colors = count_colors_used(current_coloring)
                    if current_num_colors < best_num_colors:
                        best_num_colors = current_num_colors
                        best_coloring = current_coloring.copy()
                    found = True
                    break
            if not found:
                break
    
    # Return best coloring found
    return best_coloring.tolist()
# EVOLVE-BLOCK-END

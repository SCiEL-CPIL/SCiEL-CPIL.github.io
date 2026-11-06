import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def is_2_colorable(n, edges):
        """Check if hypergraph is 2-colorable using backtracking with unit propagation."""
        if not edges:
            return True
        
        # Build adjacency: for each edge, which other edges share vertices
        # Use bitmask representation for edges
        m = len(edges)
        if m == 0:
            return True
        
        # Precompute for each edge, the bitmask of edges that conflict (share a vertex)
        # Actually, for 2-colorability: an edge is monochromatic if all its vertices get same color
        # We need: no edge is monochromatic in the final coloring
        
        # Fast approach: try all colorings up to symmetry? No, too slow for n>10
        # Use backtracking with unit propagation
        
        # For each edge, store the vertices
        # State: assignment of vertices to colors (0, 1, or -1 for unassigned)
        # Constraint: each edge must have at least one 0 and at least one 1
        
        # Better: use the fact that we can fix vertex 0 to color 0 (symmetry breaking)
        # Then do DFS
        
        # Even faster: use a SAT-like approach
        # Variables: x_v for each vertex v (0 or 1)
        # Clause for each edge: at least one vertex is 0 AND at least one vertex is 1
        # i.e., NOT(all 0) AND NOT(all 1)
        # NOT(all 0) = OR over vertices in edge of (x_v = 1)
        # NOT(all 1) = OR over vertices in edge of (x_v = 0)
        
        # This is a 2-SAT problem? No, the clauses can be long.
        # But we can use DPLL with unit propagation.
        
        # Simpler: just try to find a valid coloring via backtracking
        # with the constraint that each edge is not monochromatic
        
        # Build list of edges as tuples
        edge_list = [tuple(e) for e in edges]
        
        # For efficiency, use bitmasks for vertices
        # n <= 26, so use Python integers as bitmasks
        
        # Assign vertices one by one, checking constraints
        # Use a smarter order: process vertices that appear in many edges first
        
        # Count vertex frequencies
        freq = [0] * n
        for e in edge_list:
            for v in e:
                freq[v] += 1
        
        # Sort vertices by frequency descending
        order = sorted(range(n), key=lambda v: -freq[v])
        
        # For each edge, track: count of 0s, count of 1s assigned so far
        # An edge is "dead" if all assigned vertices are same color and count == len(edge)
        # An edge forces remaining vertices to be the other color if count of one color == len(edge)-1
        
        # Use recursion with memoization? Might be too complex.
        # Just do DFS with pruning.
        
        def dfs(idx, colors, edge_zeros, edge_ones):
            # colors: list of -1, 0, 1 for each vertex
            # edge_zeros[i]: number of vertices in edge i assigned color 0
            # edge_ones[i]: number of vertices in edge i assigned color 1
            
            # Check if any edge is already monochromatic
            for i, e in enumerate(edge_list):
                if edge_zeros[i] == len(e) and edge_ones[i] == 0:
                    return False  # all zeros, invalid
                if edge_ones[i] == len(e) and edge_zeros[i] == 0:
                    return False  # all ones, invalid
            
            # Unit propagation: if an edge has all but one vertex assigned same color,
            # the last vertex must be the other color
            for i, e in enumerate(edge_list):
                remaining = [v for v in e if colors[v] == -1]
                if len(remaining) == 1:
                    v = remaining[0]
                    if edge_zeros[i] == len(e) - 1 and edge_ones[i] == 0:
                        # Must be 1
                        if colors[v] != -1:
                            if colors[v] != 1:
                                return False
                        else:
                            colors[v] = 1
                            edge_ones[i] += 1
                            # Propagate to other edges containing v
                            for j, ej in enumerate(edge_list):
                                if v in ej and j != i:
                                    edge_ones[j] += 1
                    elif edge_ones[i] == len(e) - 1 and edge_zeros[i] == 0:
                        # Must be 0
                        if colors[v] != -1:
                            if colors[v] != 0:
                                return False
                        else:
                            colors[v] = 0
                            edge_zeros[i] += 1
                            for j, ej in enumerate(edge_list):
                                if v in ej and j != i:
                                    edge_zeros[j] += 1
                elif len(remaining) == 0:
                    # All assigned, check if monochromatic
                    if edge_zeros[i] == len(e) or edge_ones[i] == len(e):
                        return False
            
            if idx == len(order):
                # All vertices assigned, check all edges
                for i, e in enumerate(edge_list):
                    if edge_zeros[i] == 0 or edge_ones[i] == 0:
                        return False
                return True
            
            v = order[idx]
            
            # Try color 0
            old_colors = colors[v]
            old_ez = []
            old_eo = []
            affected = []
            for i, e in enumerate(edge_list):
                if v in e:
                    old_ez.append(edge_zeros[i])
                    old_eo.append(edge_ones[i])
                    affected.append(i)
            
            colors[v] = 0
            for i in affected:
                edge_zeros[i] += 1
            
            if dfs(idx + 1, colors, edge_zeros, edge_ones):
                return True
            
            # Backtrack
            colors[v] = old_colors
            for k, i in enumerate(affected):
                edge_zeros[i] = old_ez[k]
                edge_ones[i] = old_eo[k]
            
            # Try color 1
            old_ez2 = []
            old_eo2 = []
            affected2 = []
            for i, e in enumerate(edge_list):
                if v in e:
                    old_ez2.append(edge_zeros[i])
                    old_eo2.append(edge_ones[i])
                    affected2.append(i)
            
            colors[v] = 1
            for i in affected2:
                edge_ones[i] += 1
            
            if dfs(idx + 1, colors, edge_zeros, edge_ones):
                return True
            
            # Backtrack
            colors[v] = old_colors
            for k, i in enumerate(affected2):
                edge_zeros[i] = old_ez2[k]
                edge_ones[i] = old_eo2[k]
            
            return False
        
        colors = [-1] * n
        edge_zeros = [0] * len(edge_list)
        edge_ones = [0] * len(edge_list)
        
        # Fix vertex 0 to color 0 for symmetry breaking
        colors[0] = 0
        for i, e in enumerate(edge_list):
            if 0 in e:
                edge_zeros[i] += 1
        
        # Start DFS from the second vertex in order (skip vertex 0 if it's first)
        start_idx = 0
        if order[0] == 0:
            start_idx = 1
        
        return dfs(start_idx, colors, edge_zeros, edge_ones)

    # Use n=9 with K_9^5 as starting point
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    num_edges = len(all_edges)

    # Precompute edge membership for fast checking
    edge_set = set(map(tuple, all_edges))
    
    def count_bits(m):
        return bin(m).count('1')

    # Greedy reduction
    current_mask = (1 << num_edges) - 1
    
    # Verify initial is non-colorable
    current_edges = [all_edges[i] for i in range(num_edges) if current_mask & (1 << i)]
    if is_2_colorable(n, current_edges):
        # Should not happen for K_9^5
        return {'n': n, 'edges': current_edges}

    best_local_mask = current_mask
    best_local_count = num_edges

    # Multiple restarts with different strategies
    restarts = 0
    while time.time() < deadline - 3:
        restarts += 1
        curr = best_local_mask
        
        # Strategy 1: Random greedy removal
        indices = list(range(num_edges))
        random.shuffle(indices)
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for i in indices:
                if time.time() > deadline - 3:
                    break
                if curr & (1 << i):
                    new_mask = curr & ~(1 << i)
                    test_edges = [all_edges[j] for j in range(num_edges) if new_mask & (1 << j)]
                    if not is_2_colorable(n, test_edges):
                        curr = new_mask
                        changed = True
        
        count = count_bits(curr)
        if count < best_local_count:
            best_local_count = count
            best_local_mask = curr
        
        if best_local_count <= 51:
            break
        
        # Strategy 2: Try removing edges in order of lowest "importance"
        if time.time() < deadline - 3:
            curr = best_local_mask
            # Compute edge importance: number of other edges sharing vertices
            edge_importance = [0] * num_edges
            for i in range(num_edges):
                for j in range(i+1, num_edges):
                    if set(all_edges[i]) & set(all_edges[j]):
                        edge_importance[i] += 1
                        edge_importance[j] += 1
            
            sorted_indices = sorted(range(num_edges), key=lambda i: edge_importance[i])
            for i in sorted_indices:
                if time.time() > deadline - 3:
                    break
                if curr & (1 << i):
                    new_mask = curr & ~(1 << i)
                    test_edges = [all_edges[j] for j in range(num_edges) if new_mask & (1 << j)]
                    if not is_2_colorable(n, test_edges):
                        curr = new_mask
            
            count = count_bits(curr)
            if count < best_local_count:
                best_local_count = count
                best_local_mask = curr

    best_edges = [all_edges[i] for i in range(num_edges) if best_local_mask & (1 << i)]
    
    return {'n': n, 'edges': best_edges}

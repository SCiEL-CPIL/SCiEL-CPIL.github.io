"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _build_adj_matrix(edges, n):
    """Build a numpy adjacency matrix from a list of edges."""
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    return adj

def _count_c6s_fast(colors, n):
    """Count total number of C6s across all color classes using numpy."""
    total = 0
    # Build adjacency matrices for each color
    adj_mats = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_mats[c][i, j] = True
            adj_mats[c][j, i] = True
            idx += 1
    
    for c in range(4):
        adj = adj_mats[c]
        # Count C6s by counting paths of length 5 between pairs (u,v) where u<v
        # A C6 is a cycle v0-v1-v2-v3-v4-v5-v0
        # We can count by: for each pair (a,b), count paths of length 4 from a to b
        # that don't revisit a or b, then check if (a,b) is an edge
        # Actually, simpler: count all 6-cycles by enumerating 6-tuples with constraints
        
        # Use the fact that C6 count = (1/12) * sum over all ordered 6-tuples forming a cycle
        # But that's expensive. Let's use a smarter approach.
        
        # Count C6s by: for each vertex v, count pairs of neighbors (a,b) of v
        # such that there's a path of length 4 from a to b not using v.
        # This is getting complex. Let's just do a direct count with numpy optimization.
        
        # For n <= 20, we can afford a moderately efficient count.
        # Count by: for each 6-subset, check if it forms a C6 in the subgraph.
        # There are C(n,6) subsets. For each, check if the induced subgraph is a C6.
        
        # Even simpler: count all simple cycles of length 6 by backtracking with pruning.
        # But we want speed. Let's use the following:
        # Number of C6s = (1/2) * sum_{(u,v) in E} (number of paths of length 4 from u to v not using u or v internally)
        # Wait, that's not right either. A C6 has 6 edges, so if we fix an edge (u,v),
        # the rest of the cycle is a path of length 4 from u to v (or v to u) that doesn't use u or v internally.
        # But each C6 is counted 6 times this way (once for each edge), and each edge can be traversed in 2 directions...
        # Actually, let's just count directly.
        
        # For n <= 20, let's do a direct enumeration with some numpy help.
        # Count C6s by: for each starting vertex s, do a DFS to find paths of length 5
        # where all vertices are distinct and the last vertex is adjacent to s.
        # To avoid overcounting, require s to be the smallest vertex in the cycle.
        
        count = 0
        for s in range(n):
            # DFS from s, only visiting vertices > s (except s itself)
            # Path length 5 (6 vertices total)
            # We'll use iterative DFS with a stack for speed
            
            # For each neighbor of s (with vertex > s), start a path
            neighbors = [v for v in range(s+1, n) if adj[s, v]]
            
            # DFS: current vertex, path so far
            # We need paths of length 4 more (total 5 edges, 6 vertices)
            # Let's use recursion with numpy for neighbor lookup
            
            def dfs_count(current, path_len, path):
                if path_len == 5:
                    # Check if path[0] (which is s) is adjacent to current
                    if adj[s, current]:
                        return 1
                    return 0
                
                total = 0
                for nxt in range(s+1, n):
                    if nxt not in path and adj[current, nxt]:
                        path.append(nxt)
                        total += dfs_count(nxt, path_len + 1, path)
                        path.pop()
                return total
            
            for start_neighbor in neighbors:
                path = [start_neighbor]
                count += dfs_count(start_neighbor, 1, path)
        
        total += count
    
    return total

def _has_c6_fast(colors, n):
    """Check if any color class contains a C6."""
    return _count_c6s_fast(colors, n) > 0

def _random_local_search(initial_colors, n, max_steps=50000, deadline=None):
    """Perform local search to minimize C6 count."""
    colors = list(initial_colors)
    best_colors = list(colors)
    best_c6_count = _count_c6s_fast(colors, n)
    
    if best_c6_count == 0:
        return colors
    
    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge to flip
        idx = random.randint(0, len(colors) - 1)
        old_color = colors[idx]
        new_color = random.choice([c for c in range(4) if c != old_color])
        colors[idx] = new_color
        
        new_c6_count = _count_c6s_fast(colors, n)
        
        if new_c6_count < best_c6_count:
            best_c6_count = new_c6_count
            best_colors = list(colors)
            if best_c6_count == 0:
                return colors
        else:
            # Revert
            colors[idx] = old_color
        
        step += 1
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Start with the known n=9 solution
    n0 = 9
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    best_n = n0
    best_colors = list(colors0)
    
    # Try to extend from n=9 up to n=17
    for target_n in range(n0 + 1, 18):
        if time.time() > deadline - 15:
            break
        
        # Initialize new colors: construct list in lexicographic order
        # For pairs (i,j) where j < n0, use the color from colors0.
        # Otherwise, use a random color.
        colors = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if j < n0:
                    # Calculate index of (i,j) in the lexicographic list for K_n0
                    idx0 = i * (n0 - 1) - i * (i - 1) // 2 + (j - i - 1)
                    colors.append(colors0[idx0])
                else:
                    colors.append(random.randint(0, 3))
        
        # Run local search
        result = _random_local_search(colors, target_n, max_steps=10000, deadline=deadline - 5)
        
        if _count_c6s_fast(result, target_n) == 0:
            best_n = target_n
            best_colors = result
        else:
            # If we can't find a valid coloring for this n, try a few more random restarts
            # before giving up on this n
            found = False
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                # Re-initialize with random colors for the new vertices
                colors = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if j < n0:
                            idx0 = i * (n0 - 1) - i * (i - 1) // 2 + (j - i - 1)
                            colors.append(colors0[idx0])
                        else:
                            colors.append(random.randint(0, 3))
                
                result = _random_local_search(colors, target_n, max_steps=5000, deadline=deadline - 5)
                if _count_c6s_fast(result, target_n) == 0:
                    best_n = target_n
                    best_colors = result
                    found = True
                    break
            
            if not found:
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

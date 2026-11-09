import time
import random
import numpy as np

def _has_mono_k222(adj, n):
    """Check if adjacency matrix contains a monochromatic K2,2,2."""
    # K2,2,2 has 6 vertices: {a,b} x {c,d} x {e,f}
    # All 12 edges between the three pairs must be same color
    # For efficiency, iterate over pairs of vertices for the first pair
    # and check if there exist two disjoint pairs in the common neighborhood
    
    # Precompute: for each pair (i,j), the common neighbors of same color
    # adj[i][j] = 1 if edge (i,j) is color 1, 0 if color 0
    
    # For a monochromatic K2,2,2 in color c:
    # We need vertices a,b,c,d,e,f such that all edges between {a,b},{c,d},{e,f} have color c
    # This means a,b are both connected to c,d,e,f with color c
    # and c,d are both connected to e,f with color c
    # and a,b are connected to each other with color c
    
    # Efficient approach: for each pair (a,b) with adj[a][b]==c,
    # find common neighbors with color c, then check if among those
    # there exist two disjoint pairs.
    
    # Actually, let's use a different approach:
    # For each triple of disjoint pairs, check if all 12 edges are same color.
    # But that's O(n^6) which is too slow for n=30.
    
    # Better: for each pair (a,b), compute common neighbors in color 0 and color 1.
    # For color c: common_c = {v : adj[a][v]==c and adj[b][v]==c}
    # Then we need to find two disjoint pairs (c,d) and (e,f) in common_c
    # such that adj[c][d]==c and adj[e][f]==c.
    
    # Even better: for each pair (a,b) with adj[a][b]==c,
    # let S = {v != a,b : adj[a][v]==c and adj[b][v]==c}
    # Then we need to find a matching of size 2 in the subgraph induced by S with color c.
    # I.e., find c,d,e,f in S all distinct such that adj[c][d]==c and adj[e][f]==c.
    
    # For n=30, |S| <= 28. We can check this efficiently.
    
    for a in range(n):
        for b in range(a+1, n):
            for c_color in (0, 1):
                if adj[a][b] != c_color:
                    continue
                # Find common neighbors with color c_color
                S = []
                for v in range(n):
                    if v == a or v == b:
                        continue
                    if adj[a][v] == c_color and adj[b][v] == c_color:
                        S.append(v)
                if len(S) < 4:
                    continue
                # Check if there exist two disjoint edges in S with color c_color
                m = len(S)
                # Find all edges in S with color c_color
                edges = []
                for i in range(m):
                    for j in range(i+1, m):
                        if adj[S[i]][S[j]] == c_color:
                            edges.append((i, j))
                # Check if two disjoint edges exist
                found = False
                for i in range(len(edges)):
                    if found:
                        break
                    a1, b1 = edges[i]
                    for j in range(i+1, len(edges)):
                        a2, b2 = edges[j]
                        if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                            found = True
                            break
                if found:
                    return True
    return False

def _is_valid_coloring(colors, n):
    """Check if a coloring is valid (no monochromatic K2,2,2)."""
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
    return not _has_mono_k222(adj, n)

def _count_mono_k222(adj, n):
    """Count the number of monochromatic K2,2,2 subgraphs."""
    count = 0
    for a in range(n):
        for b in range(a+1, n):
            for c_color in (0, 1):
                if adj[a][b] != c_color:
                    continue
                S = []
                for v in range(n):
                    if v == a or v == b:
                        continue
                    if adj[a][v] == c_color and adj[b][v] == c_color:
                        S.append(v)
                if len(S) < 4:
                    continue
                m = len(S)
                edges = []
                for i in range(m):
                    for j in range(i+1, m):
                        if adj[S[i]][S[j]] == c_color:
                            edges.append((i, j))
                for i in range(len(edges)):
                    a1, b1 = edges[i]
                    for j in range(i+1, len(edges)):
                        a2, b2 = edges[j]
                        if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # First, try to find a valid n=30 coloring via local search
    # Start from a random initialization
    n_target = 30
    num_edges = n_target * (n_target - 1) // 2
    
    # Try multiple random restarts
    best_n = 29
    best_colors_29 = None
    
    # Generate n=29 Paley coloring as fallback
    n29 = 29
    squares = set()
    for x in range(1, n29 // 2 + 1):
        squares.add((x * x) % n29)
    colors29 = []
    for i in range(n29):
        for j in range(i + 1, n29):
            diff = (j - i) % n29
            if diff in squares:
                colors29.append(0)
            else:
                colors29.append(1)
    best_colors_29 = colors29
    
    # Try to find n=30 coloring
    random.seed(42)
    
    max_restarts = 50
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        # Build adjacency matrix
        adj = [[0]*n_target for _ in range(n_target)]
        idx = 0
        for i in range(n_target):
            for j in range(i+1, n_target):
                adj[i][j] = colors[idx]
                adj[j][i] = colors[idx]
                idx += 1
        
        # Local search: flip edges to reduce monochromatic K2,2,2 count
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            # Check if valid
            if not _has_mono_k222(adj, n_target):
                # Found a valid coloring!
                final_colors = []
                idx = 0
                for i in range(n_target):
                    for j in range(i+1, n_target):
                        final_colors.append(adj[i][j])
                        idx += 1
                return {'n': n_target, 'colors': final_colors}
            
            # Pick a random edge to flip
            edge_idx = random.randint(0, num_edges - 1)
            i, j = None, None
            cnt = 0
            for a in range(n_target):
                for b in range(a+1, n_target):
                    if cnt == edge_idx:
                        i, j = a, b
                        break
                    cnt += 1
                if i is not None:
                    break
            
            # Try flipping
            old_color = adj[i][j]
            new_color = 1 - old_color
            adj[i][j] = new_color
            adj[j][i] = new_color
            colors[edge_idx] = new_color
            
            # Check if improvement (simple acceptance for now)
            # For efficiency, just accept and continue
            
        # After local search, if not valid, try another restart
    
    # If we didn't find n=30, return n=29
    return {'n': best_n, 'colors': best_colors_29}

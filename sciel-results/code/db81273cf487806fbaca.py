import time
import random

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s margin for 120s limit
    
    # Base construction for n=12
    n_base = 12
    colors = []
    
    # Part 1: Edges within 0..10
    for i in range(11):
        for j in range(i + 1, 11):
            d = j - i
            c = min(d, 11 - d) - 1
            colors.append(c)
            
    # Part 2: Edges (11, u) for u=0..10
    # Constraint: c(u) != c(x) if x-u = +/- 2 mod 11
    # c(u) = u % 5 works
    for u in range(11):
        colors.append(u % 5)
        
    current_n = 12
    current_colors = colors
    
    # Try to extend to larger n
    # We will represent the graph as a list of lists of colors
    # adj[i][j] = color, -1 if not connected
    # But we need to check C4s efficiently.
    # For each color k, we need to know if adding edge (new, u) with color k
    # creates a C4.
    # This happens if there is a v already connected to new with color k
    # such that u and v have a common neighbor in color k.
    
    # Precompute common neighbors for each color in the current graph
    # common_neigh[k][u][v] = True if u,v have common neighbor in color k
    
    def build_common_neigh(n, col_list):
        # col_list is flat list of colors for i<j
        # Reconstruct adjacency
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = col_list[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        common = [[None]*n for _ in range(5)]
        for k in range(5):
            common[k] = [[False]*n for _ in range(n)]
            # For each pair u,v, check if they share a neighbor w in color k
            # w is neighbor of u in k if adj[u][w]==k
            # This is O(n^3 * 5) which is fine for n=20
            for u in range(n):
                for v in range(u+1, n):
                    # Check all w
                    has_common = False
                    for w in range(n):
                        if w != u and w != v:
                            if adj[u][w] == k and adj[v][w] == k:
                                has_common = True
                                break
                    common[k][u][v] = has_common
                    common[k][v][u] = has_common
        return adj, common

    adj, common = build_common_neigh(current_n, current_colors)
    
    best_n = current_n
    best_colors = current_colors[:]
    
    # Greedy extension
    next_vertex = current_n
    while next_vertex < 29 and time.time() - start_time < time_limit:
        # Try to add vertex next_vertex
        assigned = [-1] * next_vertex # colors for (next_vertex, u)
        # We process u from 0 to next_vertex-1
        # To make it deterministic or randomized? Randomized with restarts is better.
        
        def try_add_vertex(v_idx, adj, common, n_current):
            # v_idx is the index of the new vertex (n_current)
            # We need to assign colors to (v_idx, u) for u=0..n_current-1
            # assigned[u] will be the color
            assigned = [-1] * n_current
            # We can use backtracking
            # But simple greedy might fail. Let's try greedy first.
            # Order of u: random? Or 0..n-1?
            # Let's try 0..n-1
            for u in range(n_current):
                possible_colors = []
                for k in range(5):
                    # Check if color k is valid for (v_idx, u)
                    # It is invalid if there exists w < u (already assigned)
                    # such that assigned[w] == k and common[k][u][w] is True
                    valid = True
                    for w in range(u):
                        if assigned[w] == k and common[k][u][w]:
                            valid = False
                            break
                    if valid:
                        possible_colors.append(k)
                
                if not possible_colors:
                    return None # Failed
                # Heuristic: pick color with fewest existing neighbors
                # or just first available
                # Let's pick the one that minimizes future constraints
                # For now, just pick first available
                assigned[u] = possible_colors[0]
            
            return assigned
        
        # Try multiple times with randomization if it fails
        success = False
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Randomize the order of u or the choice of color
            # Simple: just try the greedy. If it fails, break and try next n?
            # No, if it fails for this n, we can't add this vertex.
            # But maybe a different assignment works.
            # Let's add a bit of randomness
            random.shuffle(range(next_vertex)) # Not used directly in simple greedy above
            
            # Let's implement a slightly smarter greedy:
            # Pick u with fewest options first? (MRV)
            # This requires checking all u for all colors.
            
            assigned = [-1] * next_vertex
            # We'll use a simple backtracking for small n
            # But for n=20, backtracking might be slow.
            # Let's stick to greedy with random tie-breaking.
            
            for u in range(next_vertex):
                possible_colors = []
                for k in range(5):
                    valid = True
                    for w in range(u):
                        if assigned[w] == k and common[k][u][w]:
                            valid = False
                            break
                    if valid:
                        possible_colors.append(k)
                
                if not possible_colors:
                    break # Failed this attempt
                # Random choice among possible
                assigned[u] = random.choice(possible_colors)
            
            if assigned[-1] != -1: # All assigned
                # Verify the whole graph? 
                # The greedy check ensures no C4 with the new vertex.
                # Old C4s are already gone.
                # So it is valid.
                
                # Update adj and common
                # Append new row/col to adj
                new_adj = [row[:] for row in adj]
                for i in range(next_vertex):
                    new_adj.append([-1] * (next_vertex + 1))
                    new_adj[i].append(assigned[i])
                    new_adj[next_vertex][i] = assigned[i]
                new_adj[next_vertex][next_vertex] = -1
                
                # Update colors list
                # We need to append colors for (next_vertex, u) in lex order
                # Lex order: (0, next_vertex), (1, next_vertex), ...
                # Wait, the output format is lexicographic (i,j) with i<j.
                # So (0, n), (1, n), ..., (n-1, n) are at the end.
                new_colors = current_colors + assigned
                
                current_n = next_vertex + 1
                current_colors = new_colors
                adj, common = build_common_neigh(current_n, current_colors)
                
                best_n = current_n
                best_colors = current_colors[:]
                
                success = True
                break # Move to next vertex
            
        if not success:
            # Could not add this vertex
            # Maybe try a different starting point? No, we are stuck.
            # Return best found so far.
            break
            
        next_vertex += 1
        
    return {'n': best_n, 'colors': best_colors}

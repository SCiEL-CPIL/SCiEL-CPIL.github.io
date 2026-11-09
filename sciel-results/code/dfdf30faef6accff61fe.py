"""Greedy randomized construction for C4-free 5-coloring of K_n."""
import time
import random

def _is_valid_coloring(n, colors):
    """Check that no color class contains a C4."""
    # colors is a list of length n*(n-1)//2 in lexicographic order
    # Build adjacency sets per color
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            adj[c].add((i, j))
    
    # For each color, check no two vertices have two common neighbors
    for c in range(5):
        # Build neighbor lists
        neigh = [set() for _ in range(n)]
        for (u, v) in adj[c]:
            neigh[u].add(v)
            neigh[v].add(u)
        for i in range(n):
            for j in range(i + 1, n):
                if len(neigh[i] & neigh[j]) >= 2:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with n=11 (known valid) and try to extend
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to build larger colorings using greedy randomized approach
    for target_n in range(12, 29):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + target_n)
            
            # Start from the known n=11 coloring and try to add vertices
            # Actually, let's try a fresh greedy construction for the target size
            colors = [-1] * (target_n * (target_n - 1) // 2)
            
            # Greedy: assign colors to edges in order, choosing the color that
            # creates the fewest new "dangerous" pairs
            # Track for each color and each pair of vertices, how many common neighbors they have
            # A pair (u,v) in color c is "dangerous" if they have exactly 1 common neighbor in color c
            # We want to avoid creating situations where adding an edge would complete a C4
            
            # For efficiency, track common-neighbor counts
            # cn[c][u][v] = number of common neighbors of u and v in color c
            # This is O(5 * n^2) memory which for n=29 is 5*841 = 4205, fine
            
            cn = [[[0]*target_n for _ in range(target_n)] for _ in range(5)]
            
            valid = True
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    idx = i * target_n - i * (i+1) // 2 + j - i - 1
                    # Actually let me recompute index properly
                    # Number of edges before (i,j): sum_{k=0}^{i-1} (target_n - 1 - k) + (j - i - 1)
                    # = i*(target_n-1) - i*(i-1)//2 + j - i - 1
                    # Let me just use a simpler formula
                    idx = 0
                    for k in range(i):
                        idx += target_n - 1 - k
                    idx += j - i - 1
                    
                    # Try each color, pick the one that minimizes maximum cn increment
                    best_color = -1
                    best_score = float('inf')
                    
                    for c in range(5):
                        # If we add edge (i,j) in color c, then for all neighbors u of i in color c,
                        # cn[c][u][j] increases by 1, and for all neighbors v of j in color c,
                        # cn[c][i][v] increases by 1
                        # The dangerous thing is if any cn[c][u][j] or cn[c][i][v] becomes 2
                        # (meaning u and j would have 2 common neighbors in color c)
                        
                        # Count how many would become exactly 2 (creating a C4)
                        danger = 0
                        # Check neighbors of i in color c
                        for u in range(target_n):
                            if u == i or u == j:
                                continue
                            # Is (i,u) an edge in color c? Check if already assigned
                            idx_iu = 0
                            a, b = min(i, u), max(i, u)
                            for k in range(a):
                                idx_iu += target_n - 1 - k
                            idx_iu += b - a - 1
                            if colors[idx_iu] == c:
                                # u is a neighbor of i in color c
                                if cn[c][u][j] == 1:
                                    danger += 1  # would become 2
                        # Check neighbors of j in color c
                        for v in range(target_n):
                            if v == i or v == j:
                                continue
                            idx_jv = 0
                            a, b = min(j, v), max(j, v)
                            for k in range(a):
                                idx_jv += target_n - 1 - k
                            idx_jv += b - a - 1
                            if colors[idx_jv] == c:
                                if cn[c][i][v] == 1:
                                    danger += 1
                        
                        if danger == 0:
                            # No immediate C4 creation, pick this color
                            best_color = c
                            best_score = 0
                            break
                        elif danger < best_score:
                            best_score = danger
                            best_color = c
                    
                    if best_color == -1:
                        valid = False
                        break
                    
                    colors[idx] = best_color
                    c = best_color
                    
                    # Update cn: for all u neighbor of i in color c, cn[c][u][j] += 1
                    # and cn[c][j][u] += 1
                    for u in range(target_n):
                        if u == i or u == j:
                            continue
                        idx_iu = 0
                        a, b = min(i, u), max(i, u)
                        for k in range(a):
                            idx_iu += target_n - 1 - k
                        idx_iu += b - a - 1
                        if colors[idx_iu] == c:
                            cn[c][u][j] += 1
                            cn[c][j][u] += 1
                        # Also for v neighbor of j
                    for v in range(target_n):
                        if v == i or v == j:
                            continue
                        idx_jv = 0
                        a, b = min(j, v), max(j, v)
                        for k in range(a):
                            idx_jv += target_n - 1 - k
                        idx_jv += b - a - 1
                        if colors[idx_jv] == c:
                            cn[c][i][v] += 1
                            cn[c][v][i] += 1
                            # Also cn[c][i][v] already counted above? No, this is different pair
                            # Wait, I need to be more careful
                            # When adding edge (i,j) in color c:
                            # For each u with edge (i,u) in color c: cn[c][u][j] += 1
                            # For each v with edge (j,v) in color c: cn[c][i][v] += 1
                            # These are different updates unless u==v which can't happen
                            # But I'm double-counting in the loop above. Let me restructure.
                    
                    if not valid:
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify
                if _is_valid_coloring(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break
        
        if not found:
            # Couldn't extend to target_n, try next
            continue
    
    return {'n': best_n, 'colors': best_colors}

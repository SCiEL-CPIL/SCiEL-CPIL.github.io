import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Construct the base K32 coloring from the seed
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    n_base = 32
    colors = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                colors.append(log[i ^ j] % 3)
            else:
                colors.append(3)
                
    # Helper to get color of edge (u, w) in current graph
    # We maintain a list of colors. 
    # Index for (u, w) with u < w:
    # In the full list of size n(n-1)/2, the index is sum_{k=0}^{u-1} (n-k-1) + (w-u-1)
    # This is expensive to calculate on the fly. 
    # Better to maintain a 2D array or a dictionary for the current coloring.
    
    # Let's use a 2D array for speed, or a flat array with a helper.
    # For n up to 60, a 2D list of lists is fine.
    # Initialize adj_color matrix
    adj_color = [[-1] * n_base for _ in range(n_base)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = colors[idx]
            adj_color[i][j] = c
            adj_color[j][i] = c
            idx += 1
            
    current_n = n_base
    
    # Function to extend by 1 vertex
    def try_extend(n, adj_color, deadline):
        """
        Tries to extend the coloring of K_n to K_{n+1}.
        Returns (success, new_adj_color, new_colors_list)
        """
        # Variables: x[0] ... x[n-1] representing color of edge (n, u)
        # Constraints: For each c, the set {u | x[u]==c} must be independent in G_c.
        # G_c edges are (u, w) where adj_color[u][w] == c.
        
        # Precompute conflict masks for each vertex u and color c.
        # mask[u][c] is a bitmask of vertices w such that adj_color[u][w] == c.
        # If x[u] = c, then x[w] cannot be c for any w in mask[u][c].
        
        masks = [[0] * 4 for _ in range(n)]
        for u in range(n):
            for w in range(n):
                if u != w:
                    c = adj_color[u][w]
                    masks[u][c] |= (1 << w)
                    
        # Backtracking solver
        assignment = [-1] * n
        # Order vertices by heuristic: most constrained? 
        # All vertices have 4 options initially.
        # Maybe order by degree in the "conflict" graph?
        # Let's just use 0..n-1 for now, or sort by number of 0s in masks?
        
        # To speed up, we can use bitmasks for domains.
        # domain[u] is a bitmask of allowed colors? No, colors are 0-3.
        # Better: maintain a list of available colors for each u?
        
        # Standard backtracking with forward checking
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
                
            u = idx
            # Try colors 0, 1, 2, 3
            # Check if color c is valid for u
            # Valid if no previously assigned v has x[v]==c and u in masks[v][c]
            # Or equivalently, v in masks[u][c] and x[v]==c
            
            for c in range(4):
                valid = True
                for v in range(idx):
                    if assignment[v] == c:
                        if (masks[u][c] >> v) & 1:
                            valid = False
                            break
                if valid:
                    assignment[u] = c
                    if backtrack(idx + 1):
                        return True
                    assignment[u] = -1
            return False
            
        if backtrack(0):
            # Build new adj_color
            new_adj_color = [row[:] for row in adj_color]
            new_adj_color.append([0] * (n + 1))
            for i in range(n):
                new_adj_color[i].append(assignment[i])
                new_adj_color[n][i] = assignment[i]
            new_adj_color[n][n] = 0 # self loop, irrelevant
            
            # Build new colors list
            # We need to append edges (n, 0), (n, 1), ..., (n, n-1) to the list?
            # The problem statement says: "one colour ... for each pair (i,j), i<j ... in lexicographic order"
            # The list is flat.
            # The new edges are (n, 0) ... (n, n-1).
            # In lexicographic order of pairs (i,j) with i<j:
            # The previous list ended with pairs involving n-1.
            # The new pairs are (0, n), (1, n), ..., (n-1, n).
            # These come after all pairs (i, j) with j < n.
            # So we just append assignment[0] ... assignment[n-1] to the colors list.
            
            new_colors = []
            # Copy old colors
            new_colors.extend(colors[:]) # Wait, 'colors' is the global list from seed. 
            # We should maintain a local colors list.
            
            return True, new_adj_color, assignment
        else:
            return False, None, None

    # Main loop: try to extend
    # We need to maintain the flat 'colors' list correctly.
    # The seed's 'colors' list is for K32.
    # When we add vertex 32, we append 32 colors.
    # When we add vertex 33, we append 33 colors.
    
    # Let's re-initialize colors from adj_color to be safe, or just append.
    # The seed's colors list is correct for K32.
    current_colors = colors[:]
    
    while current_n < 62:
        if time.time() > deadline - 10: # Reserve time
            break
            
        success, new_adj, assignment = try_extend(current_n, adj_color, deadline)
        if success:
            # Update state
            adj_color = new_adj
            # Append the new edge colors to current_colors
            current_colors.extend(assignment)
            current_n += 1
        else:
            break
            
    return {'n': current_n, 'colors': current_colors}

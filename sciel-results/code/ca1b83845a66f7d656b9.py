import time
import random

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    # Reconstruct adjacency or just check common neighbors
    # For efficiency in small n, we can just check all 4-tuples
    # But n will be small (<= 20 maybe), so O(n^4) is fine for verification.
    # However, we use incremental checks during construction.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    best_n = 1
    best_colors = []
    
    # Try to build up from n=1
    # We can use the known n=11 solution as a base if we want to jump start
    # But let's try to build from scratch with randomization to see if we can reach higher.
    # Actually, the n=11 cyclic solution is a good base.
    # Let's initialize with n=11 cyclic solution.
    
    n_init = 11
    colors_init = [min(j - i, n_init - j + i) - 1 for i in range(n_init) for j in range(i + 1, n_init)]
    best_n = n_init
    best_colors = colors_init
    
    # Helper to get color of edge (u, v) from list
    def get_color(colors, n, u, v):
        if u > v: u, v = v, u
        # Index calculation
        # Sum of 1 to u-1 is u(u-1)/2
        # Then add (v - u)
        # Wait, lexicographic order:
        # (0,1), (0,2)...(0,n-1), (1,2)...
        # Index for (u, v) where u < v:
        # count = sum_{k=0}^{u-1} (n - 1 - k) + (v - u - 1) ?
        # Let's just use a 2D array for internal state during construction.
        pass

    # Internal state: 2D list colors_matrix[n][n]
    # Initialize with n=11
    n = n_init
    cm = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            cm[i][j] = colors_init[idx]
            cm[j][i] = colors_init[idx]
            idx += 1
    
    # Precompute common neighbors counts for each color
    # common[c][u][v] = number of common neighbors of u and v in color c
    # This is expensive to update.
    # Alternative: For each color c, maintain adjacency lists adj_c[u].
    # Check if adding edge (u, v) in color c creates a C4.
    # This happens if there is a w such that (u, w) and (v, w) are in color c.
    # i.e., w in adj_c[u] and w in adj_c[v].
    
    def try_add_vertex():
        nonlocal n, cm, best_n, best_colors
        if time.time() > deadline:
            return False
        
        new_n = n + 1
        # Extend cm
        new_cm = [[-1]*new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(n):
                new_cm[i][j] = cm[i][j]
        
        # We need to assign colors to edges (new_n-1, i) for i in 0..n-1
        # Let's use backtracking
        # Order of assignment: 0 to n-1
        # To speed up, we can use a heuristic: assign colors that are "least constrained"
        
        # State: colors_assigned[0..n-1]
        # We maintain adj_c[u] for c in 0..4, u in 0..new_n-1
        # Initially, adj_c[u] is from cm for u, v < n.
        
        adj = [[[] for _ in range(new_n)] for _ in range(5)]
        for i in range(n):
            for j in range(i+1, n):
                c = cm[i][j]
                adj[c][i].append(j)
                adj[c][j].append(i)
        
        # Function to check if we can assign color c to edge (new_v, u)
        def can_assign(u, c):
            # Check if u and any existing neighbor of new_v in color c form a C4
            # Neighbors of new_v in color c so far:
            # We are assigning sequentially, so we only need to check against previously assigned neighbors of new_v in color c.
            # Let prev_neighbors = [v for v in assigned if color(new_v, v) == c]
            # For each v in prev_neighbors, check if u and v have a common neighbor in color c.
            # Common neighbor w must be in adj[c][u] and adj[c][v].
            # We can just iterate through adj[c][u] and check if w is in prev_neighbors? No.
            # w must be a common neighbor of u and v in the EXISTING graph (vertices 0..n-1).
            # So we check if adj[c][u] and adj[c][v] intersect.
            # To make this fast, we can precompute intersection sizes or use sets.
            # Given small n, sets are fine.
            
            # Actually, we can just check:
            # Is there a w in adj[c][u] such that w is in adj[c][v]?
            # And w must be in 0..n-1.
            
            # Optimization:
            # If we maintain sets for adj[c][u], intersection is fast.
            pass

        # Let's implement a simple backtracking with sets for adj lists.
        # Convert adj lists to sets for fast intersection
        adj_sets = [[set(adj[c][u]) for u in range(new_n)] for c in range(5)]
        
        # assigned_colors[i] for i in 0..n-1
        assigned = [-1]*n
        
        # Order of vertices to assign: 0, 1, ..., n-1
        # We can randomize the order to escape local minima
        order = list(range(n))
        random.shuffle(order)
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                # Success
                for i in range(n):
                    c = assigned[i]
                    new_cm[new_n-1][i] = c
                    new_cm[i][new_n-1] = c
                    adj_sets[c][new_n-1].add(i)
                    adj_sets[c][i].add(new_n-1)
                return True
            
            u = order[idx]
            # Try each color
            # Heuristic: try colors that are less frequent for u?
            for c in range(5):
                # Check validity
                # For all v assigned color c previously, check if u and v have common neighbor in c
                valid = True
                for v in range(n):
                    if assigned[v] == c:
                        # Check intersection of adj_sets[c][u] and adj_sets[c][v]
                        if adj_sets[c][u] & adj_sets[c][v]:
                            valid = False
                            break
                if valid:
                    assigned[u] = c
                    # Update adj sets for new_v (conceptually)
                    # We don't actually update adj_sets for new_v yet, 
                    # because the check `adj_sets[c][u] & adj_sets[c][v]` 
                    # relies on the EXISTING graph structure.
                    # The new vertex new_v is not in adj_sets yet.
                    # The condition is: u and v must not have a common neighbor in the EXISTING graph.
                    # If they do, adding new_v as a common neighbor makes 2 common neighbors.
                    # So the check is correct.
                    
                    if backtrack(idx + 1):
                        return True
                    
                    assigned[u] = -1
            
            return False
        
        if backtrack(0):
            n = new_n
            cm = new_cm
            # Update best if better
            if n > best_n:
                best_n = n
                # Flatten cm to list
                best_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        best_colors.append(cm[i][j])
            return True
        return False

    # Try to add vertices
    # We can try multiple times if it fails, with different shuffles
    attempts = 0
    while time.time() < deadline and n < 29:
        if not try_add_vertex():
            attempts += 1
            if attempts > 50:
                # Restart with different random seed or break
                # Maybe try a different strategy?
                # For now, just break to save time
                break
        else:
            attempts = 0 # Reset attempts on success

    # Return best found
    return {'n': best_n, 'colors': best_colors}

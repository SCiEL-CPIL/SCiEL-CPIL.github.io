import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def is_valid_coloring(n, colors):
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[c][u] & adj[c][v]
                    if common.bit_count() >= 2:
                        return False
        return True

    def try_extend(n_prev, parent_colors):
        # Returns colors for n_prev + 1 if successful
        n = n_prev + 1
        # Reconstruct adj for n_prev
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n_prev):
            for j in range(i + 1, n_prev):
                c = parent_colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        v = n_prev
        # New colors list
        new_colors = parent_colors[:]
        
        # We need to assign colors to (v, u) for u in 0..v-1
        # We will use a randomized backtracking search
        
        # Precompute conflict constraints?
        # For each color c, if we assign u to c, we block all x such that adj[c][u] & adj[c][x] != 0.
        
        # Let's use a simple recursive backtracking
        # State: 
        #   u: current vertex to assign (0..v-1)
        #   assigned_colors: list of colors for (v, 0)..(v, u-1)
        #   current_adj_v: bitmask of neighbors of v in each color (so far)
        #   Actually, we just need to check if adding u to color c creates a C4.
        #   C4 check: If we add edge (v,u) with color c, does it create a C4?
        #   It creates a C4 if there exists x < u such that (v,x) is also color c, 
        #   and u, x have a common neighbor w in color c (in the old graph).
        #   So, for a fixed color c, the set of vertices assigned to c so far, S_c, 
        #   must be such that for any u, x in S_c, adj[c][u] & adj[c][x] == 0.
        
        # This is exactly the independent set condition.
        
        # We can implement a greedy backtracking:
        # 1. Shuffle u from 0 to v-1.
        # 2. For each u, try colors 0..4.
        # 3. If color c is valid (u doesn't conflict with existing S_c), assign it.
        # 4. If no color works, backtrack.
        
        # To make it faster, we can pre-compute for each u and c, the list of x that conflict with u in color c.
        # conflict_mask[c][u] = bitmask of x such that adj[c][u] & adj[c][x] != 0.
        
        conflict_masks = [[0] * n for _ in range(5)]
        for c in range(5):
            for u in range(n_prev):
                mask = 0
                # u conflicts with x if they share a neighbor in color c
                # This is equivalent to: adj[c][u] & adj[c][x] != 0
                # We can compute this by iterating x?
                # Or: The set of neighbors of u in color c is N_u.
                # Any x that is a neighbor of a neighbor of u in color c is a conflict?
                # No. x conflicts with u if there is w such that (u,w)_c and (x,w)_c.
                # So x is in the union of neighbors of neighbors of u?
                # Actually, `adj[c][u] & adj[c][x]` is the set of common neighbors.
                # So we just need to check if this intersection is non-empty.
                
                # Efficient way:
                # For each w in adj[c][u]:
                #   for each x in adj[c][w]:
                #     if x != u: conflict_masks[c][u] |= (1 << x)
                # This is O(k^2) per u. k is small.
                
                neighbors_u = adj[c][u]
                while neighbors_u:
                    lsb = neighbors_u & -neighbors_u
                    w = lsb.bit_length() - 1
                    neighbors_u -= lsb
                    # Add all neighbors of w in color c to conflict mask of u
                    # Exclude u itself? If x=u, it's not a pair.
                    # adj[c][w] contains u. We should mask out u.
                    conflict_masks[c][u] |= (adj[c][w] & ~(1 << u))
        
        # Now backtrack
        # We assign colors to vertices 0..v-1
        # We maintain S_c as bitmasks
        S = [0] * 5
        
        def backtrack(u_idx, order):
            if u_idx == v:
                return True
            
            u = order[u_idx]
            for c in range(5):
                # Check if u can be added to S[c]
                # u must not conflict with any x in S[c]
                # Conflict if conflict_masks[c][u] & S[c] != 0
                if (conflict_masks[c][u] & S[c]) == 0:
                    S[c] |= (1 << u)
                    # Assign color c to edge (v, u)
                    # We need to store this in new_colors
                    # Index for (v, u) where v > u
                    # In lexicographic order (i,j) with i<j:
                    # Pairs are (0,1)..(0,n-1), (1,2)..(1,n-1) ...
                    # Index of (u, v) with u < v:
                    # sum_{k=0}^{u-1} (n-1-k) + (v-u-1)
                    # Wait, n is the new size.
                    # The pairs are (i,j) for 0<=i<j<n.
                    # The new edges are (u, v) for u in 0..v-1.
                    # These are the last v edges in the list?
                    # No.
                    # Total edges = n(n-1)/2.
                    # Edges involving v are (0,v), (1,v), ..., (v-1, v).
                    # In the flat list, these are at the end?
                    # Let's check order:
                    # i=0: j=1..n-1
                    # i=1: j=2..n-1
                    # ...
                    # i=v-1: j=v..n-1 (only j=v since n=v+1)
                    # So (v-1, v) is the last edge.
                    # (0, v) is at index: sum_{k=0}^{v-1} (n-1-k) ... no.
                    # Index of (i, j) = i*n - i*(i+1)//2 + j - i - 1
                    # Let's just append to a list and then construct the final array.
                    
                    # To avoid index calculation errors, let's just store assignments in a dict or list of length v
                    assignments[u] = c
                    
                    if backtrack(u_idx + 1, order):
                        return True
                    
                    S[c] &= ~(1 << u)
                    assignments[u] = -1
            return False
        
        assignments = [-1] * v
        order = list(range(v))
        
        # Try multiple random orders
        for _ in range(20):
            random.shuffle(order)
            S = [0] * 5
            assignments = [-1] * v
            if backtrack(0, order):
                # Success
                # Construct new_colors
                new_colors = parent_colors[:]
                # Add new edges
                # We need to insert them in the correct positions
                # It's easier to rebuild the whole colors list from scratch
                # Or just append and sort? No, the order is fixed.
                
                # Let's rebuild
                final_colors = [0] * (n * (n - 1) // 2)
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if i == v or j == v:
                            # One of them is v
                            u = i if j == v else j
                            final_colors[idx] = assignments[u]
                        else:
                            # Original edge
                            # Find index in parent_colors
                            # This is slow.
                            pass
                # Rebuilding is O(n^2). Acceptable for n=26.
                # Let's do it properly.
                
                # Map (i,j) to index
                # We can just create a 2D array of colors and flatten
                col_grid = [[-1] * n for _ in range(n)]
                p_idx = 0
                for i in range(n_prev):
                    for j in range(i + 1, n_prev):
                        c = parent_colors[p_idx]
                        col_grid[i][j] = c
                        col_grid[j][i] = c
                        p_idx += 1
                        
                for u in range(v):
                    c = assignments[u]
                    col_grid[u][v] = c
                    col_grid[v][u] = c
                
                final_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        final_colors.append(col_grid[i][j])
                        
                return final_colors
        return None

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    valid_colorings = {11: [best_colors]}
    
    for target_n in range(12, 29):
        if time.time() > deadline - 10:
            break
            
        found = False
        parents = valid_colorings.get(target_n - 1, [])
        if not parents:
            continue
            
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            parent = random.choice(parents)
            res = try_extend(target_n - 1, parent)
            if res:
                if is_valid_coloring(target_n, res):
                    valid_colorings[target_n] = res
                    best_n = target_n
                    best_colors = res
                    found = True
                    break
        
        if not found:
            # Try harder?
            pass

    return {'n': best_n, 'colors': best_colors}

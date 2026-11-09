import time
import sys
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # We will build the graph incrementally.
    # colors[i][j] for i < j.
    # We store colors in a 2D array for easy access.
    # n_max_target = 17 or 18.
    
    def _check_and_extend(n_current, colors):
        """
        Try to extend the valid coloring of K_{n_current} to K_{n_current+1}.
        Returns (success, new_colors_matrix) or (False, None).
        """
        if time.time() > deadline - 5:
            return False, None
            
        n_new = n_current + 1
        new_colors = [row[:] for row in colors]
        # Add new row and col
        for i in range(n_current):
            new_colors[i].append(0) # placeholder
        new_colors.append([0] * n_new)
        
        # Precompute P_c(u, v) for u, v in 0..n_current-1
        # P_c[u][v] is True if there is a path of length 4 in color c between u and v.
        # Path u-x-y-z-v.
        # We need to check all u, v < n_current.
        
        # To compute P_c, we can use matrix multiplication or just iterate paths.
        # Since n is small (<=18), iterating is fine.
        # P_c[u][v] = exists x, y, z such that:
        # (u,x) in E_c, (x,y) in E_c, (y,z) in E_c, (z,v) in E_c.
        
        # Let's build adjacency lists for each color in the current graph.
        adj = [[[] for _ in range(n_current)] for _ in range(4)]
        for u in range(n_current):
            for v in range(u + 1, n_current):
                c = colors[u][v]
                adj[c][u].append(v)
                adj[c][v].append(u)
                
        # Compute reachability by path of length 4.
        # dist_c[u][v] = min path length? No, just existence.
        # We can compute "path of length 4" by:
        # 1. Compute path of length 1 (adjacency)
        # 2. Compute path of length 2: u~x~v
        # 3. Compute path of length 3: u~x~y~v
        # 4. Compute path of length 4: u~x~y~z~v
        
        # Efficient way:
        # Let M_c be the adjacency matrix (boolean).
        # Path length 2: M_c @ M_c
        # Path length 3: (M_c @ M_c) @ M_c
        # Path length 4: (M_c @ M_c) @ (M_c @ M_c) ? No.
        # Path length 4 is M_c^4.
        # We just need boolean existence.
        
        P = [[[[False] * n_current for _ in range(n_current)] for _ in range(n_current)] for _ in range(4)]
        
        # For each color c
        for c in range(4):
            # Build boolean adjacency matrix
            M = [[False] * n_current for _ in range(n_current)]
            for u in range(n_current):
                for v in adj[c][u]:
                    M[u][v] = True
            
            # Compute M2 = M @ M (path length 2)
            M2 = [[False] * n_current for _ in range(n_current)]
            for u in range(n_current):
                for x in range(n_current):
                    if M[u][x]:
                        for v in range(n_current):
                            if M[x][v]:
                                M2[u][v] = True
            
            # Compute M4 = M2 @ M2 (path length 4)
            # Note: This finds paths of length exactly 4?
            # If there is a path of length 2, M2[u][v] is true.
            # If we do M2[u][x] and M2[x][v], we get a path u..x..v of length 2+2=4.
            # This includes paths that might backtrack, but we just need existence of SOME path of length 4.
            # Actually, if there is a path of length 2, there isn't necessarily a path of length 4.
            # But if M2[u][v] is true, does that imply a C6 risk?
            # The cycle is k-u-x-y-z-v-k.
            # The path u-x-y-z-v has 4 edges.
            # So yes, we need existence of a path of length 4.
            
            P[c] = [[False] * n_current for _ in range(n_current)]
            for u in range(n_current):
                for x in range(n_current):
                    if M2[u][x]:
                        for v in range(n_current):
                            if M2[x][v]:
                                P[c][u][v] = True
                                # Optimization: break? No, need to fill all.
        
        # Now, backtrack to assign colors to edges (new_vertex, i) for i in 0..n_current-1
        # new_vertex index is n_current.
        # We assign colors to (n_current, i).
        
        # Heuristic: order i by degree in the "conflict graph"?
        # Conflict for color c between i and j is P[c][i][j].
        # Let's just iterate i from n_current-1 down to 0.
        
        # State: assigned_colors[i] for i in 0..n_current-1.
        # assigned_colors[i] is the color of edge (n_current, i).
        
        assigned = [-1] * n_current
        
        def backtrack(idx):
            if time.time() > deadline - 2:
                return False
            if idx < 0:
                return True
                
            # Try all 4 colors for edge (n_current, idx)
            # To prune, check against already assigned vertices j > idx.
            for c in range(4):
                # Check if assigning color c to (n_current, idx) conflicts with any j > idx
                # where assigned[j] == c and P[c][idx][j] is True.
                valid = True
                for j in range(idx + 1, n_current):
                    if assigned[j] == c:
                        if P[c][idx][j]:
                            valid = False
                            break
                if valid:
                    assigned[idx] = c
                    if backtrack(idx - 1):
                        return True
                    assigned[idx] = -1
            return False
        
        if backtrack(n_current - 1):
            # Update new_colors
            for i in range(n_current):
                new_colors[n_current][i] = assigned[i]
                new_colors[i][n_current] = assigned[i]
            return True, new_colors
        else:
            return False, None

    # Start with n=1
    # colors is a list of lists.
    # For n=1, colors = [[0]] (dummy)
    # Actually, let's start with n=2 or n=3 to have some structure?
    # n=1 is trivial.
    
    current_colors = [[0]]
    current_n = 1
    
    # We want to maximize n.
    # Let's try to extend up to n=17.
    # If it fails, we keep the last successful one.
    
    best_n = current_n
    best_colors = current_colors
    
    # To make the search more robust, we might want to randomize the order of colors or vertices?
    # The current backtracking is deterministic.
    # If it fails for n=17, it might be because the specific choice of colors for previous vertices led to a dead end.
    # We should try multiple restarts or randomize the initial coloring?
    # But we are building from scratch.
    # We can randomize the color order in the backtracking loop.
    
    rng = np.random.RandomState(42)
    
    for target_n in range(2, 18): # Try to reach 17
        if time.time() > deadline - 10:
            break
            
        success = False
        # Try a few random permutations of color priority to escape local minima?
        # Or just try the standard order first.
        
        # Let's try standard order first.
        ok, new_c = _check_and_extend(current_n, current_colors)
        if ok:
            current_n = target_n
            current_colors = new_c
            best_n = current_n
            best_colors = current_colors
            success = True
        else:
            # If standard fails, try randomizing color order?
            # The backtracking tries 0,1,2,3.
            # We can modify _check_and_extend to accept a color_order.
            # For now, let's just break if it fails?
            # No, maybe a different previous coloring would work.
            # But we don't have backtracking on previous vertices.
            # This is a greedy extension. It might fail even if a solution exists.
            
            # To improve, we can restart the entire construction with different random seeds?
            # But construction is deterministic based on the greedy choice.
            # Let's add a "randomized greedy" option.
            
            pass

    # If we stopped early, return the best found.
    
    # Flatten colors
    result_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            result_colors.append(int(best_colors[i][j]))
            
    return {'n': best_n, 'colors': result_colors}

import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, leave 35s buffer)
    
    # Fallback: n=11 solution
    # K11 split into five Hamiltonian cycles by circular distance 1..5
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    
    best_n = n_fallback
    best_colors = colors_fallback

    # Strategy:
    # 1. Start with a valid coloring for n=11.
    # 2. Attempt to extend to n=12, 13, ... up to 26 or time limit.
    # 3. Use Backtracking with Forward Checking to solve the extension.
    #    - Variables: Edges of the new vertex (v_new, v_old) for v_old in 0..n-1.
    #    - Domain: {0, 1, 2, 3, 4} for each edge.
    #    - Constraints: 
    #      a) For any pair of existing vertices (u, v), they cannot share >1 common neighbor in any color.
    #         Adding a new vertex w with colors (w,u) and (w,v) might create a C4 if (u,v) already had 
    #         a common neighbor x in that color.
    #      b) Specifically, if we assign color c to (w, u) and color c' to (w, v):
    #         If c == c', then u and v must NOT already share a common neighbor in color c.
    #         (Because w would become a second common neighbor of u and v in color c).
    #      c) Also, we must ensure w doesn't form C4s with existing structures in other ways?
    #         A C4 involves 4 vertices. If we add w, the C4 must include w.
    #         The cycle is w - a - b - c - w.
    #         This requires edges (w,a), (a,b), (b,c), (c,w) to be same color.
    #         This is equivalent to: a and c share a common neighbor w in color k, AND a and c share 
    #         a common neighbor b in color k? No.
    #         Let's stick to the "common neighbors" definition.
    #         A graph is C4-free iff for any pair of vertices (u,v), |N(u) ∩ N(v)| <= 1.
    #         When adding w:
    #         For any pair (u, v) of OLD vertices:
    #         New common neighbors of u and v in color c are:
    #         {x in old N_c(u) ∩ N_c(v)} U {w if (w,u) is c and (w,v) is c}.
    #         So if (w,u) is c and (w,v) is c, then |old N_c(u) ∩ N_c(v)| must be 0.
    #         
    #         For pairs involving w, e.g., (w, u):
    #         Common neighbors of w and u in color c are:
    #         {v such that (w,v) is c and (u,v) is c}.
    #         Let S_c = {v | (w,v) is c}.
    #         Then common neighbors are S_c ∩ N_c(u).
    #         We need |S_c ∩ N_c(u)| <= 1 for all u and all c.
    #         This means: For each color c, the set of vertices v connected to w with color c 
    #         must be an independent set in G_c? No, it means for each u, 
    #         the number of v in S_c that are neighbors of u in G_c is at most 1.
    #         This is a strong constraint.

    # Precompute adjacency bitmasks for the current best solution
    def get_adj_bitmasks(colors, n):
        # adj[c][u] is bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    # Convert bitmask to list of neighbors
    def mask_to_list(mask, n):
        return [i for i in range(n) if (mask >> i) & 1]

    # Solve extension from n to n+1
    def try_extend(colors, n, deadline):
        # Current adjacencies
        adj = get_adj_bitmasks(colors, n)
        
        # New vertex index is n
        # Variables: x[i] for i in 0..n-1, where x[i] is color of edge (n, i)
        # Domains: 0..4
        
        # Precompute for each color c and each vertex u:
        # The "forbidden" pairs?
        
        # Constraint 1: For any u, v in 0..n-1, if x[u] == x[v] == c, 
        # then adj[c][u] & adj[c][v] must be 0.
        # Let's precompute for each color c, a list of pairs (u,v) that are "incompatible" 
        # for sharing color c. i.e., they already share a neighbor in color c.
        incompatible = [set() for _ in range(5)]
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u] & adj[c][v]:
                        incompatible[c].add((u, v))
        
        # Constraint 2: For any u in 0..n-1, and any color c:
        # Let S_c = {v | x[v] == c}
        # We need |S_c ∩ N_c(u)| <= 1.
        # This is harder to precompute as a simple pair constraint because it depends on the set S_c.
        # However, during backtracking, we can check this incrementally.
        
        # Backtracking with MRV (Minimum Remaining Values)
        # Since all domains are size 5 initially, we need a good heuristic.
        # Let's order vertices by degree in the "incompatible" graph?
        # Or just random order?
        
        # Let's use a recursive backtracking solver.
        # To avoid recursion limit and overhead, we might use iterative or simple recursion with n<=29 (depth 29 is fine).
        
        x = [-1] * n # x[i] is color of (n, i)
        
        # Heuristic: Order vertices by "constraint tightness".
        # A vertex u is more constrained if it has many neighbors in each color.
        # Let's sort vertices by sum of degrees in all colors?
        # Actually, let's just try a few random orderings or a static one.
        
        # We will use a simple DFS with forward checking.
        
        # State:
        # assigned: list of assigned x values
        # For each color c, maintain S_c (set of vertices assigned color c)
        # For each color c and vertex u, maintain count_c_u = |S_c ∩ N_c(u)|
        
        S = [set() for _ in range(5)]
        # counts[c][u] = number of neighbors of u in color c that are in S_c
        counts = [[0] * n for _ in range(5)]
        
        # Precompute neighbor lists for quick access
        neighbors = [[mask_to_list(adj[c][u], n) for u in range(n)] for c in range(5)]
        
        def is_valid_assignment(idx, color):
            # Check if assigning x[idx] = color is valid
            # 1. Check Constraint 1: For all v < idx (already assigned? No, we assign in order 0..n-1)
            #    Actually, we assign sequentially. 
            #    If we assign x[idx] = color, we must check against all previously assigned x[v].
            #    If x[v] == color, then (idx, v) must not be in incompatible[color].
            
            # 2. Check Constraint 2: 
            #    For all u, counts[color][u] + (1 if idx in N_color(u) else 0) must be <= 1.
            #    Wait, if we add idx to S_color, then for any neighbor u of idx in color, 
            #    counts[color][u] increases by 1.
            #    We need new_counts[color][u] <= 1 for all u.
            #    So for all u in N_color(idx), counts[color][u] must be 0.
            
            # Check 1: Incompatible pairs
            for v in range(idx):
                if x[v] == color:
                    if (min(idx, v), max(idx, v)) in incompatible[color]:
                        return False
            
            # Check 2: Neighbor counts
            for u in neighbors[color][idx]:
                if counts[color][u] > 0:
                    return False
            
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            # Try colors 0-4
            # Heuristic: Try colors that are "least used" or "most constrained"?
            # Let's just try 0-4.
            for color in range(5):
                if is_valid_assignment(idx, color):
                    # Assign
                    x[idx] = color
                    S[color].add(idx)
                    # Update counts
                    old_counts = []
                    for u in neighbors[color][idx]:
                        old_counts.append((u, counts[color][u]))
                        counts[color][u] += 1
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Unassign
                    x[idx] = -1
                    S[color].remove(idx)
                    for u, oc in old_counts:
                        counts[color][u] = oc
            
            return False

        # To improve performance, we can shuffle the order of vertices or colors?
        # If we fix the order 0..n-1, we might hit dead ends early.
        # Let's try a few random permutations of the vertex indices if the first one fails.
        
        # Actually, for n=12, depth is 11. It's small.
        # For n=26, depth is 25.
        
        # Let's run backtracking.
        if backtrack(0):
            # Success
            new_colors = colors[:]
            # Append colors for edges (n, 0), (n, 1), ... (n, n-1)
            # Note: The lexicographic order for edges in K_{n+1} is:
            # All edges of K_n first, then (0, n), (1, n), ... (n-1, n)?
            # No, lexicographic order on pairs (i, j) with i < j.
            # Pairs involving n: (0, n), (1, n), ..., (n-1, n).
            # These come AFTER all pairs (i, j) where j < n.
            # So we just append x[0], x[1], ..., x[n-1].
            new_colors.extend(x)
            return new_colors, n + 1
        else:
            return None, n

    # Main loop: Try to extend
    current_colors = best_colors[:]
    current_n = best_n
    
    # We know n=11 is valid. Try to go to 12, 13...
    # If we fail at n, we can try a different random restart of the backtracking?
    # Or just accept the failure and return best found.
    # Given the structure, if n=12 fails, n=13 will likely fail too unless we change the base.
    # But let's try to extend as far as possible.
    
    for target_n in range(current_n + 1, 29):
        if time.time() > deadline - 10:
            break
            
        new_colors, new_n = try_extend(current_colors, current_n, deadline)
        
        if new_colors is not None:
            current_colors = new_colors
            current_n = new_n
            best_n = current_n
            best_colors = current_colors[:]
            
            # If we reached 26, we are at the known bound. 
            # Try to go to 27?
            # The task says 27 or 28 would be new.
            # If we get to 26, we have a huge win (11 -> 26).
            # Let's keep trying.
        else:
            # Failed to extend
            # Try a few random restarts of the extension process?
            # The backtracking is deterministic given the order.
            # We can randomize the color order or vertex order.
            # Let's implement a randomized version of try_extend if we have time.
            
            restarts = 5
            success = False
            for r in range(restarts):
                if time.time() > deadline - 5:
                    break
                # Randomize vertex order?
                # This requires modifying try_extend to accept an order.
                # For simplicity, let's just re-run try_extend with a slight modification:
                # Shuffle the colors in the domain? No, colors are symmetric.
                # Shuffle the vertex indices in the backtracking?
                
                # Let's just break and return best. 
                # Extending from a failed state is hard without restarting from scratch 
                # or using a different heuristic.
                break
            
            if not success:
                break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

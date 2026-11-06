import time
import random

def _has_k222(colors, n):
    """
    Check if the coloring contains a monochromatic K2,2,2.
    colors: list of 0/1 for edges (i,j) with i<j.
    n: number of vertices.
    Returns True if a monochromatic K2,2,2 exists.
    """
    # Build adjacency sets for each color
    # adj[color][u] is a set of vertices v such that edge (u,v) has that color
    adj = [set() for _ in range(n)]
    adj_c = [[set() for _ in range(n)] for _ in range(2)]
    
    # Map edge index to (u, v)
    # index = i*n - i*(i+1)//2 + (j-i-1) ? No, simpler to just iterate
    # We can build a 2D matrix or just use the list
    # Let's build a 2D matrix for speed O(1) lookup
    # mat[i][j] = color
    mat = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            mat[i][j] = c
            mat[j][i] = c
            idx += 1
            
    for i in range(n):
        for j in range(i+1, n):
            c = mat[i][j]
            adj_c[c][i].add(j)
            adj_c[c][j].add(i)
            
    # Check for K2,2,2
    # Iterate over all pairs (u, v)
    for u in range(n):
        for v in range(u+1, n):
            for c in [0, 1]:
                # Common neighbors in color c
                common = adj_c[c][u].intersection(adj_c[c][v])
                # Remove u and v just in case (they are not in their own adj)
                # common contains w such that (u,w) and (v,w) are color c
                
                # We need to find a K2,2 in the subgraph induced by 'common' using color c
                # K2,2 is 4 vertices a,b,c,d with edges ab, ac, ad, bc, bd, cd? No.
                # K2,2 has parts {a,b} and {c,d}. Edges are a-c, a-d, b-c, b-d.
                # So we need two disjoint pairs {a,b} and {c,d} in 'common'
                # such that all 4 edges between them are color c.
                
                # Efficient check:
                # For each pair {a, b} in common:
                #   Check if there is another pair {c, d} in common \ {a,b}
                #   such that c is connected to a and b in color c, and d is connected to a and b in color c.
                #   This is equivalent to: 
                #   Let S = common
                #   For each pair {a,b} in S:
                #       N_a = adj_c[c][a].intersection(S)
                #       N_b = adj_c[c][b].intersection(S)
                #       Candidates = N_a.intersection(N_b)
                #       If |Candidates| >= 2, then we have a K2,2 (a,b) and any two from Candidates.
                #       Wait, K2,2 requires a,b to be in one part and c,d in the other.
                #       The edges must be between the parts.
                #       So yes, if we have a,b and two other vertices x,y that are connected to BOTH a and b in color c,
                #       then {a,b} and {x,y} form a K2,2 in color c?
                #       No, the edges must be between {a,b} and {x,y}.
                #       We checked x is connected to a and b.
                #       We checked y is connected to a and b.
                #       Do we need to check edges between x and y? No, K2,2 is bipartite.
                #       Do we need to check edges within {a,b} or {x,y}? No.
                #       So yes, this logic holds.
                
                S_list = list(common)
                s_len = len(S_list)
                if s_len < 4:
                    continue
                    
                # Optimization: Precompute intersection sets?
                # For small s_len, just loop.
                for i_idx in range(s_len):
                    a = S_list[i_idx]
                    for j_idx in range(i_idx+1, s_len):
                        b = S_list[j_idx]
                        # Find vertices in S connected to both a and b in color c
                        # We can just check each vertex in S
                        # Actually, we just need to find 2 such vertices.
                        count = 0
                        # Iterate over S again? O(s^3) per pair.
                        # s is max 27. 27^3 = 19683.
                        # Pairs (u,v) is 406. 406 * 2 * 19683 ~ 16 million ops.
                        # This might be too slow in Python if done repeatedly in local search.
                        # But for the initial check, it's fine.
                        
                        # Better:
                        # N_a_in_S = adj_c[c][a] & common
                        # N_b_in_S = adj_c[c][b] & common
                        # Common2 = N_a_in_S & N_b_in_S
                        # If len(Common2) >= 2, we found a K2,2.
                        # Note: a and b are in common, so they are in S.
                        # a is connected to b? If edge (a,b) is color c, then b is in N_a.
                        # But we need x,y distinct from a,b.
                        # So Common2 should exclude a and b.
                        
                        n_a = adj_c[c][a] & common
                        n_b = adj_c[c][b] & common
                        common2 = n_a & n_b
                        # Remove a and b
                        if a in common2: common2.remove(a)
                        if b in common2: common2.remove(b)
                        
                        if len(common2) >= 2:
                            return True
    return False

def _paley_coloring(n):
    """
    Generate Paley coloring for prime n = 4k+1.
    """
    if n % 4 != 1:
        return None
    squares = {x*x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try n=29 first
    n = 29
    colors = _paley_coloring(n)
    
    if colors is not None:
        if not _has_k222(colors, n):
            return {'n': n, 'colors': colors}
            
    # If Paley 29 fails, try local search on 29
    # Start from Paley 29 or random
    best_colors = colors
    best_violations = 10**9 # Just a flag, _has_k222 is boolean
    
    # We need a count function for local search
    def count_violations(colors_list, n_val):
        # Re-use logic but return count
        adj_c = [[set() for _ in range(n_val)] for _ in range(2)]
        mat = [[-1]*n_val for _ in range(n_val)]
        idx = 0
        for i in range(n_val):
            for j in range(i+1, n_val):
                c = colors_list[idx]
                mat[i][j] = c
                mat[j][i] = c
                adj_c[c][i].add(j)
                adj_c[c][j].add(i)
                idx += 1
        
        count = 0
        for u in range(n_val):
            for v in range(u+1, n_val):
                for c in [0, 1]:
                    common = adj_c[c][u].intersection(adj_c[c][v])
                    S_list = list(common)
                    s_len = len(S_list)
                    if s_len < 4:
                        continue
                    # Check for K2,2 in S_list
                    # We count distinct K2,2 structures? Or just boolean?
                    # For local search, boolean is enough if we just want 0.
                    # But to guide search, count is better.
                    # However, counting all K2,2s is hard.
                    # Let's just return 1 if any found, 0 else?
                    # No, we need a gradient.
                    # Let's count the number of "bad" pairs (u,v) that have a K2,2 in their common neighborhood.
                    # This is an upper bound on violations.
                    
                    found = False
                    for i_idx in range(s_len):
                        a = S_list[i_idx]
                        n_a = adj_c[c][a] & common
                        for j_idx in range(i_idx+1, s_len):
                            b = S_list[j_idx]
                            n_b = adj_c[c][b] & common
                            common2 = n_a & n_b
                            if a in common2: common2.remove(a)
                            if b in common2: common2.remove(b)
                            if len(common2) >= 2:
                                count += 1
                                found = True
                                break
                        if found:
                            break
        return count

    current_colors = colors[:]
    current_cost = count_violations(current_colors, n)
    
    if current_cost == 0:
        return {'n': n, 'colors': current_colors}
        
    # Local Search
    # Simulated Annealing or Simple Hill Climbing with Random Restarts
    # Since n=29 is small, we can afford many steps.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        if restart == 0:
            curr = current_colors[:]
        else:
            # Random restart
            curr = [random.choice([0,1]) for _ in range(len(current_colors))]
            
        curr_cost = count_violations(curr, n)
        if curr_cost == 0:
            return {'n': n, 'colors': curr}
            
        temp = 1.0
        for step in range(5000):
            if time.time() > deadline:
                break
            if curr_cost == 0:
                return {'n': n, 'colors': curr}
                
            # Pick random edge
            edge_idx = random.randint(0, len(curr)-1)
            old_c = curr[edge_idx]
            new_c = 1 - old_c
            curr[edge_idx] = new_c
            
            new_cost = count_violations(curr, n)
            
            if new_cost < curr_cost:
                curr_cost = new_cost
            else:
                # Accept with probability exp(-delta/temp)
                delta = new_cost - curr_cost
                if delta == 0:
                    curr_cost = new_cost
                else:
                    prob = pow(2.71828, -delta/temp)
                    if random.random() > prob:
                        # Reject
                        curr[edge_idx] = old_c
                    else:
                        curr_cost = new_cost
            
            temp *= 0.995 # Cooling
            
    # If 29 failed, try 28?
    # But 29 is the record, so it should work.
    # If we run out of time, return the best we have?
    # The problem asks to maximize n.
    # If we can't get 29, we should return a valid smaller n.
    # But we don't have a valid smaller n guaranteed.
    # We can construct a valid n=17 (Paley).
    # Let's fall back to n=17 if 29 fails.
    
    colors_17 = _paley_coloring(17)
    if colors_17 and not _has_k222(colors_17, 17):
        return {'n': 17, 'colors': colors_17}
        
    # Fallback: n=1
    return {'n': 1, 'colors': []}

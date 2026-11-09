"""Two-colourings without a monochromatic K2,2,2 (R(K2,2,2))"""
import time
import random
from itertools import combinations

def has_k222(adj, n):
    """Check if adjacency matrix adj (list of lists, adj[i][j]=1 if edge exists) contains K2,2,2."""
    # K2,2,2 has 6 vertices. We need to find 6 vertices {a,b,c,d,e,f}
    # such that a,b are connected to c,d,e,f (all 4) in the same color,
    # and c,d,e,f form K4 in that color.
    # Actually K2,2,2: partition into {a,b}, {c,d,e,f} where all edges between
    # the two parts exist, and within {c,d,e,f} all edges exist (K4).
    # Wait, K2,2,2 is the complete tripartite graph K_{2,2,2}.
    # It has 3 parts of size 2: {a,b}, {c,d}, {e,f}
    # Edges: all between different parts, none within same part.
    # So we need vertices a,b,c,d,e,f where:
    #   a-b: NO edge, c-d: NO edge, e-f: NO edge
    #   a-c, a-d, a-e, a-f, b-c, b-d, b-e, b-f, c-e, c-f, d-e, d-f: ALL edges
    
    # For each pair of non-adjacent vertices (a,b), check if their common neighbors
    # contain a K2,2 (i.e., two disjoint pairs of non-adjacent vertices with all cross edges)
    
    n_nonadj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                n_nonadj[i][j] = 1
                n_nonadj[j][i] = 1
    
    # Precompute common neighbor sets for non-adjacent pairs
    # For speed, use bitmasks
    adj_mask = [0]*n
    nonadj_mask = [0]*n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)
            else:
                if i != j:
                    nonadj_mask[i] |= (1 << j)
    
    # For each non-adjacent pair (a,b), common neighbors = adj_mask[a] & adj_mask[b]
    # Among common neighbors, we need to find c,d,e,f where c-d non-adj, e-f non-adj,
    # and all cross edges exist between {c,d} and {e,f}
    # i.e., c,e,c,f,d,e,d,f all edges, and c-d no edge, e-f no edge
    
    for a in range(n):
        for b in range(a+1, n):
            if adj[a][b]:
                continue
            # common neighbors of a and b
            cn = adj_mask[a] & adj_mask[b]
            # iterate over non-adjacent pairs (c,d) within cn
            cn_list = [v for v in range(n) if (cn >> v) & 1]
            for ci in range(len(cn_list)):
                c = cn_list[ci]
                for di in range(ci+1, len(cn_list)):
                    d = cn_list[di]
                    if adj[c][d]:
                        continue
                    # c and d are non-adjacent common neighbors
                    # Now need e,f in cn, e!=c,d, f!=c,d, e-f non-adj,
                    # and e,f connected to both c and d
                    # e must be in cn (connected to a,b) AND connected to c and d
                    # f must be in cn AND connected to c and d
                    # and e-f non-adjacent
                    # e,f distinct from c,d
                    candidates = []
                    for v in cn_list:
                        if v == c or v == d:
                            continue
                        if adj[c][v] and adj[d][v]:
                            candidates.append(v)
                    # find non-adjacent pair in candidates
                    for ei in range(len(candidates)):
                        e = candidates[ei]
                        for fi in range(ei+1, len(candidates)):
                            f = candidates[fi]
                            if not adj[e][f]:
                                return True
    return False

def has_k222_fast(adj, n):
    """Faster check using bitmasks."""
    adj_mask = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    all_mask = (1 << n) - 1
    
    for a in range(n):
        for b in range(a+1, n):
            if adj_mask[a] & (1 << b):
                continue  # a,b adjacent, skip
            cn = adj_mask[a] & adj_mask[b]
            cn &= ~(1 << a)
            cn &= ~(1 << b)
            if cn == 0:
                continue
            # Get list of common neighbors
            cn_list = []
            m = cn
            while m:
                v = (m & -m).bit_length() - 1
                cn_list.append(v)
                m &= m - 1
            
            # For each non-adjacent pair (c,d) in cn_list
            for ci in range(len(cn_list)):
                c = cn_list[ci]
                for di in range(ci+1, len(cn_list)):
                    d = cn_list[di]
                    if adj_mask[c] & (1 << d):
                        continue  # c,d adjacent
                    # Need e,f in cn_list, e!=c,d, f!=c,d
                    # e,f connected to both c and d, e-f non-adjacent
                    cand_mask = cn & adj_mask[c] & adj_mask[d]
                    cand_mask &= ~(1 << c)
                    cand_mask &= ~(1 << d)
                    cand_list = []
                    m2 = cand_mask
                    while m2:
                        v = (m2 & -m2).bit_length() - 1
                        cand_list.append(v)
                        m2 &= m2 - 1
                    for ei in range(len(cand_list)):
                        e = cand_list[ei]
                        for fi in range(ei+1, len(cand_list)):
                            f = cand_list[fi]
                            if not (adj_mask[e] & (1 << f)):
                                return True
    return False

def validate_colors(n, colors):
    """Validate that neither color class contains K2,2,2."""
    adj0 = [[0]*n for _ in range(n)]
    adj1 = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == 0:
                adj0[i][j] = 1
                adj0[j][i] = 1
            else:
                adj1[i][j] = 1
                adj1[j][i] = 1
            idx += 1
    return not has_k222(adj0, n) and not has_k222(adj1, n)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    best_n = 17
    best_colors = None
    
    # Start with Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    p17 = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    if validate_colors(17, p17):
        best_n = 17
        best_colors = p17[:]
    
    # Try to extend using local search / greedy + backtracking
    # Strategy: start from a valid coloring, try to add vertices one by one
    # with random initial assignments, then repair conflicts
    
    rng = random.Random(42)
    
    def try_extend(base_n, base_colors, target_n, max_time):
        """Try to extend a coloring from base_n to target_n vertices."""
        t0 = time.time()
        best_local = None
        best_local_n = base_n
        
        # Greedy extension with backtracking
        def greedy_extend(start_n, start_colors, target_n):
            """Greedy: add vertices one at a time, assign colors to avoid K2,2,2."""
            n = start_n
            colors = start_colors[:]
            adj0 = [[0]*target_n for _ in range(target_n)]
            adj1 = [[0]*target_n for _ in range(target_n)]
            
            idx = 0
            for i in range(start_n):
                for j in range(i+1, start_n):
                    if colors[idx] == 0:
                        adj0[i][j] = 1
                        adj0[j][i] = 1
                    else:
                        adj1[i][j] = 1
                        adj1[j][i] = 1
                    idx += 1
            
            while n < target_n:
                if time.time() > max_time:
                    break
                new_v = n
                found = False
                for attempt in range(200):
                    # Try a random assignment for new vertex
                    assignment = []
                    for i in range(n):
                        assignment.append(rng.randint(0, 1))
                    
                    # Check if this assignment creates K2,2,2 in either color
                    ok = True
                    # For color 0: new edges from new_v to existing vertices
                    # Check if adding these creates K2,2,2
                    # Simple check: for each pair (a,b) of existing vertices
                    # that are non-adjacent in color 0, check if new_v connects to both
                    # and then we'd have a K2,2,2
                    # This is complex; use the full check
                    
                    # Build temp adjacency
                    adj0_new = [row[:] for row in adj0]
                    adj1_new = [row[:] for row in adj1]
                    for i in range(n):
                        adj0_new[new_v][i] = adj0_new[i][new_v] = assignment[i]
                        adj1_new[new_v][i] = adj1_new[i][new_v] = 1 - assignment[i]
                    
                    if has_k222(adj0_new, n+1) or has_k222(adj1_new, n+1):
                        continue
                    
                    # Valid! Add it
                    for i in range(n):
                        if assignment[i] == 0:
                            adj0[i][new_v] = 1
                            adj0[new_v][i] = 1
                        else:
                            adj1[i][new_v] = 1
                            adj1[new_v][i] = 1
                        colors.append(assignment[i])
                    n += 1
                    found = True
                    break
                
                if not found:
                    break
            
            return n, colors, adj0, adj1
        
        # Try multiple random starts
        for restart in range(50):
            if time.time() > max_time:
                break
            t_check = time.time()
            n, colors, adj0, adj1 = greedy_extend(base_n, base_colors, target_n)
            if n > best_local_n:
                best_local_n = n
                best_local = colors[:]
            if n >= target_n:
                break
        
        return best_local_n, best_local
    
    # Try extending from 17 to larger n
    for target in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        if time.time() > deadline - 10:
            break
        t_limit = min(deadline - 5, time.time() + 10)
        n_found, colors_found = try_extend(17, p17, target, t_limit)
        if colors_found and n_found > best_n:
            if validate_colors(n_found, colors_found):
                best_n = n_found
                best_colors = colors_found
                print(f"Extended to n={best_n}")
        if time.time() > deadline - 5:
            break
    
    # If we found a larger coloring, try to extend further
    if best_n > 17 and best_colors:
        for target in range(best_n + 1, min(best_n + 5, 30)):
            if time.time() > deadline - 10:
                break
            t_limit = min(deadline - 5, time.time() + 8)
            n_found, colors_found = try_extend(best_n, best_colors, target, t_limit)
            if colors_found and n_found > best_n:
                if validate_colors(n_found, colors_found):
                    best_n = n_found
                    best_colors = colors_found
                    print(f"Extended to n={best_n}")
            if time.time() > deadline - 5:
                break
    
    # Final: try random restarts with different seeds for more exploration
    if time.time() < deadline - 15:
        for seed in [1, 2, 3, 7, 13]:
            if time.time() > deadline - 10:
                break
            rng2 = random.Random(seed)
            # Try building from scratch with random init + repair
            target = max(best_n + 1, 20)
            t_limit = min(deadline - 5, time.time() + 8)
            
            # Random initial coloring for target vertices
            n_try = target
            colors_try = [rng2.randint(0, 1) for _ in range(n_try * (n_try - 1) // 2)]
            
            # Repair: iteratively fix violations
            # This is complex; skip for now and use greedy from small base
            pass
    
    if best_colors is None:
        best_colors = p17
    
    return {'n': best_n, 'colors': best_colors}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        adj[u].add(v)
        adj[v].add(u)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    m = len(edges)
    if m == 0:
        return []

    # Map edge to bit position
    edge_to_bit = {}
    for i, (u, v) in enumerate(edges):
        edge_to_bit[(u, v)] = 1 << i
        edge_to_bit[(v, u)] = 1 << i

    # Generate maximal bicliques
    # A biclique is (L, R). 
    # We generate candidates by expanding each edge to its maximal biclique.
    candidates = []
    seen_bicliques = set()
    
    # Helper to get bitmask of edges in a biclique
    def get_biclique_mask(L, R):
        mask = 0
        for u in L:
            for v in R:
                if (u, v) in edge_to_bit:
                    mask |= edge_to_bit[(u, v)]
        return mask

    # Helper to check if biclique is valid (all cross edges exist)
    # Since we generate from maximal expansion, they should be valid.
    # But we need to ensure L and R are disjoint? 
    # "nonempty disjoint sides"
    # In a simple graph, if u in L and u in R, then (u,u) would be an edge? 
    # The problem says "disjoint sides".
    # My expansion: L starts with {u}, R starts with {v}.
    # If u is added to R later, it means N(u) superset R_old.
    # If u is in R, and u is in L, then (u,u) must be an edge? 
    # Usually graphs are simple (no self-loops). 
    # The expansion logic:
    # L_new = {x | N(x) superset R}
    # If x is in R, then N(x) must contain all of R.
    # This implies (x, y) is an edge for all y in R.
    # If x is also in L, then (x, x) is not required to be an edge for the biclique property 
    # UNLESS the definition requires L and R to be disjoint.
    # The task says "disjoint sides".
    # So if L and R intersect, we must remove the intersection? 
    # Or does the expansion naturally prevent intersection?
    # If u in L and v in R.
    # If u is added to R, it means u is connected to all in R.
    # If v is added to L, it means v is connected to all in L.
    # If L and R intersect, say z in L ∩ R.
    # Then z is connected to all in L (since z in R) and all in R (since z in L).
    # This is fine for the biclique structure (complete bipartite graph between L and R).
    # But the task requires L and R to be disjoint.
    # If they intersect, we can just take L' = L \ R and R' = R. 
    # Wait, if z in L ∩ R, edges (z, w) for w in L \ {z} are "within-side" edges? 
    # No, (z, w) with z in R, w in L is a cross edge.
    # If z in L and z in R, then (z, z) is a self-loop? No.
    # The edges between L and R are the cross edges.
    # If L and R are not disjoint, the set of cross edges is not well-defined in the standard sense 
    # unless we assume simple graph and L ∩ R = empty.
    # The expansion might produce L and R that intersect.
    # Example: Triangle 0-1-2-0.
    # Edge (0,1). L={0}, R={1}.
    # N(0)={1,2}, N(1)={0,2}.
    # L' = {x | N(x) superset {1}} = {0, 2} (since N(2)={0,1} contains 1? No, N(2)={0,1}. 1 in N(2). Yes.)
    # Wait, N(2) = {0, 1}. Does {0,1} contain {1}? Yes.
    # So L' = {0, 2}.
    # R' = {y | N(y) superset {0,2}}.
    # N(0)={1,2} - does not contain 0.
    # N(1)={0,2} - contains 0,2. Yes.
    # N(2)={0,1} - does not contain 2.
    # So R' = {1}.
    # Next iter:
    # L'' = {x | N(x) superset {1}} = {0, 2}.
    # R'' = {y | N(y) superset {0,2}} = {1}.
    # Stable. L={0,2}, R={1}. Disjoint.
    
    # What if we have a square 0-1-2-3-0 with diagonal 0-2?
    # It seems the expansion tends to keep them disjoint if the graph is simple and no self loops.
    # Because for u to be in L and R, u must be connected to all in R (including u? No, N(u) doesn't contain u).
    # So if u in R, u is not in N(u).
    # Condition for u in L': N(u) superset R.
    # If u in R, then u in N(u) is False. So R cannot contain u if u is required to be in N(u)?
    # No, the condition is N(u) ⊇ R.
    # If u ∈ R, then u ∈ N(u) must be true for the superset to hold? 
    # No, u ∈ R means u is an element of the set R.
    # The condition is: for all v in R, (u, v) is an edge.
    # If v = u, then (u, u) must be an edge.
    # Simple graphs have no self-loops. So (u, u) is not an edge.
    # Therefore, if u ∈ R, the condition "for all v in R, (u,v) is edge" fails for v=u.
    # So u cannot be in L' if u is in R.
    # Thus L and R are always disjoint in this expansion for simple graphs.
    
    for u in range(n):
        for v in adj[u]:
            if u < v: # Avoid duplicates
                # Expand (u, v) with u in L, v in R
                L = {u}
                R = {v}
                while True:
                    L_new = set()
                    for x in range(n):
                        if adj[x].issuperset(R):
                            L_new.add(x)
                    if L_new == L:
                        break
                    L = L_new
                    R_new = set()
                    for y in range(n):
                        if adj[y].issuperset(L):
                            R_new.add(y)
                    if R_new == R:
                        break
                    R = R_new
                
                # Ensure disjoint (should be, but check)
                if L & R:
                    L = L - R
                    if not L or not R:
                        continue
                
                if L and R:
                    key = (frozenset(L), frozenset(R))
                    if key not in seen_bicliques:
                        seen_bicliques.add(key)
                        mask = get_biclique_mask(L, R)
                        if mask != 0:
                            candidates.append((L, R, mask))
                
                # Expand (v, u) with v in L, u in R
                L = {v}
                R = {u}
                while True:
                    L_new = set()
                    for x in range(n):
                        if adj[x].issuperset(R):
                            L_new.add(x)
                    if L_new == L:
                        break
                    L = L_new
                    R_new = set()
                    for y in range(n):
                        if adj[y].issuperset(L):
                            R_new.add(y)
                    if R_new == R:
                        break
                    R = R_new
                
                if L & R:
                    L = L - R
                    if not L or not R:
                        continue
                
                if L and R:
                    key = (frozenset(L), frozenset(R))
                    if key not in seen_bicliques:
                        seen_bicliques.add(key)
                        mask = get_biclique_mask(L, R)
                        if mask != 0:
                            candidates.append((L, R, mask))

    # Add single edge bicliques as fallback
    for u, v in edges:
        key = (frozenset([u]), frozenset([v]))
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            mask = get_biclique_mask([u], [v])
            candidates.append(([u], [v], mask))
        
        key = (frozenset([v]), frozenset([u]))
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            mask = get_biclique_mask([v], [u])
            candidates.append(([v], [u], mask))

    # Remove candidates that are subsets of other candidates? 
    # For exact cover, larger candidates are generally better (cover more edges).
    # But we need to be careful.
    # Let's just use all candidates.
    
    # Sort candidates by number of edges (larger first) to help greedy/backtracking
    candidates.sort(key=lambda x: bin(x[2]).count('1'), reverse=True)
    
    # Precompute for each edge, which candidates cover it
    # edge_idx -> list of candidate indices
    edge_to_candidates = [[] for _ in range(m)]
    for i, (L, R, mask) in enumerate(candidates):
        for j in range(m):
            if (mask >> j) & 1:
                edge_to_candidates[j].append(i)

    # Backtracking search
    best_solution = []
    best_count = float('inf')
    
    # Use a list to store the path
    current_path = []
    
    def search(remaining_mask, depth):
        nonlocal best_count, best_solution
        if time.time() > deadline:
            return False # Signal timeout
        
        if remaining_mask == 0:
            if depth < best_count:
                best_count = depth
                best_solution = current_path[:]
            return True
            
        if depth >= best_count:
            return False
            
        # MRV: Pick edge with fewest candidates
        # Find first set bit
        # Optimization: Precompute? No, dynamic.
        min_count = float('inf')
        target_edge = -1
        tmp_mask = remaining_mask
        while tmp_mask:
            lsb = tmp_mask & -tmp_mask
            e_idx = lsb.bit_length() - 1
            cnt = len(edge_to_candidates[e_idx])
            # Filter candidates that don't overlap with already chosen? 
            # In exact cover, we just pick one.
            if cnt < min_count:
                min_count = cnt
                target_edge = e_idx
                if min_count == 0:
                    break
            tmp_mask ^= lsb
            
        if target_edge == -1:
            return False # Should not happen if remaining_mask != 0
            
        if min_count == 0:
            return False
            
        for c_idx in edge_to_candidates[target_edge]:
            L, R, mask = candidates[c_idx]
            # Check if mask is subset of remaining_mask
            if (mask & ~remaining_mask) == 0:
                current_path.append((L, R))
                if search(remaining_mask & ~mask, depth + 1):
                    return True
                current_path.pop()
                
        return False

    initial_mask = (1 << m) - 1
    search(initial_mask, 0)
    
    # If no solution found (shouldn't happen), fallback to greedy
    if not best_solution:
        # Greedy fallback
        remaining = (1 << m) - 1
        greedy_sol = []
        while remaining:
            best_c = -1
            best_count_edges = -1
            for i, (L, R, mask) in enumerate(candidates):
                if (mask & ~remaining) == 0:
                    cnt = bin(mask & remaining).count('1')
                    if cnt > best_count_edges:
                        best_count_edges = cnt
                        best_c = i
            if best_c == -1:
                break
            L, R, mask = candidates[best_c]
            greedy_sol.append((L, R))
            remaining &= ~mask
            
        best_solution = greedy_sol
        best_count = len(best_solution)

    # Convert to required format
    result = []
    for L, R in best_solution:
        result.append([list(L), list(R)])
        
    return result

# EVOLVE-BLOCK-END

import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = P['n']
    edges_list = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges_list:
        adj[u].add(v)
        adj[v].add(u)
        
    # Canonical edge representation
    all_edges = set()
    for u, v in edges_list:
        if u > v:
            u, v = v, u
        all_edges.add((u, v))
        
    # Generate Maximal Biclique Candidates
    # For each edge, find the maximal biclique containing it
    candidates = {} # key: (frozenset(L), frozenset(R)), value: set of edges
    
    for u, v in edges_list:
        # Start with L={u}, R={v}
        L = {u}
        R = {v}
        
        # Expand to maximal
        changed = True
        while changed:
            changed = False
            # Try to expand R
            # R must be subset of intersection of neighbors of all nodes in L
            new_R = set(adj[L.pop()])
            for x in L:
                new_R = new_R.intersection(adj[x])
            # Wait, L.pop() modifies L. Need to be careful.
            # Let's use a helper to compute intersection of neighbors
            
            # Correct expansion logic:
            # R_new = { w | for all x in L, (x,w) in E }
            # L_new = { w | for all y in R, (w,y) in E }
            
            # Calculate potential new R
            if L:
                # Start with neighbors of first node in L
                first = next(iter(L))
                cand_R = set(adj[first])
                for x in L:
                    if x != first:
                        cand_R = cand_R.intersection(adj[x])
                # cand_R is the set of vertices connected to ALL nodes in L
                # We want to add vertices to R that are in cand_R but not in R
                # But we must ensure that adding them doesn't violate anything?
                # No, if w is in cand_R, it is connected to all x in L.
                # So (x, w) is an edge for all x in L.
                # Since (x, y) are edges for y in R, and (x, w) is edge,
                # the new set L x (R U {w}) is a biclique.
                # So we can add all cand_R to R?
                # Yes, R = R U cand_R.
                if not R.issuperset(cand_R):
                    R = R.union(cand_R)
                    changed = True
            else:
                R = set()
                
            # Calculate potential new L
            if R:
                first = next(iter(R))
                cand_L = set(adj[first])
                for y in R:
                    if y != first:
                        cand_L = cand_L.intersection(adj[y])
                if not L.issuperset(cand_L):
                    L = L.union(cand_L)
                    changed = True
            else:
                L = set()
                
        if not L or not R:
            continue
            
        # Ensure L and R are disjoint?
        # In a simple graph, if u in L and v in R, (u,v) is edge.
        # If u in L and u in R, (u,u) is not an edge (no self loops).
        # So L and R are naturally disjoint if graph has no self loops.
        # However, the expansion might add a node to L that was in R?
        # If w in R, is w in cand_L?
        # cand_L is neighbors of all y in R.
        # If w in R, is w a neighbor of y? Only if (w,y) is edge.
        # If w in R and y in R, (w,y) is not an edge (same side).
        # So w is NOT in adj[y].
        # So w is NOT in cand_L.
        # So L and R remain disjoint.
        
        # Generate edges for this biclique
        biclique_edges = set()
        for l in L:
            for r in R:
                if l > r:
                    biclique_edges.add((r, l))
                else:
                    biclique_edges.add((l, r))
                    
        # Filter to only valid edges in graph (should be all, by construction)
        biclique_edges = biclique_edges.intersection(all_edges)
        
        if biclique_edges:
            key = (frozenset(L), frozenset(R))
            # If we've seen this biclique, we don't need to add it again
            if key not in candidates:
                candidates[key] = biclique_edges

    # Greedy Set Cover
    uncovered = set(all_edges)
    selected = []
    
    # Sort candidates by size descending? Or just pick max coverage each time.
    # Since we re-evaluate coverage, order doesn't matter for correctness, but might for speed.
    # We will just iterate.
    
    while uncovered:
        best_candidate = None
        best_coverage = 0
        best_edges = None
        
        for key, b_edges in candidates.items():
            cov = len(b_edges.intersection(uncovered))
            if cov > best_coverage:
                best_coverage = cov
                best_candidate = key
                best_edges = b_edges
                
        if best_candidate is None or best_coverage == 0:
            # Fallback: pick any single uncovered edge
            # This shouldn't happen if candidates are maximal
            edge = next(iter(uncovered))
            u, v = edge
            selected.append( ({u}, {v}), {edge} )
            uncovered.remove(edge)
            continue
            
        selected.append( (best_candidate, best_edges) )
        uncovered.difference_update(best_edges)
        
        # Time check
        if time.time() > deadline:
            break

    # Post-processing: Try to merge bicliques
    # Check if any two selected bicliques can be merged into one valid biclique
    # This is O(k^2 * V^2) which is fine for small k
    merged = True
    while merged:
        merged = False
        new_selected = []
        i = 0
        while i < len(selected):
            curr = selected[i]
            merged_with_next = False
            if i + 1 < len(selected):
                next_b = selected[i+1]
                # Check if union is a biclique
                L1, R1 = curr[0]
                L2, R2 = next_b[0]
                
                # Check disjointness of sides in union
                # L_new = L1 U L2, R_new = R1 U R2
                # Must have L_new intersection R_new == empty
                # And all edges between L_new and R_new must exist
                
                L_new = L1.union(L2)
                R_new = R1.union(R2)
                
                if L_new.isdisjoint(R_new):
                    # Check if all cross edges exist
                    valid = True
                    for l in L_new:
                        for r in R_new:
                            e = (min(l,r), max(l,r))
                            if e not in all_edges:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Merge them
                        # Union of edges
                        union_edges = curr[1].union(next_b[1])
                        new_selected.append( ((L_new, R_new), union_edges) )
                        i += 2
                        merged = True
                        merged_with_next = True
                        continue
            
            if not merged_with_next:
                new_selected.append(curr)
                i += 1
                
        selected = new_selected
        
        if time.time() > deadline:
            break

    # Format output
    result = []
    for (L, R), _ in selected:
        # Convert to lists of Python ints
        l_list = [int(x) for x in sorted(L)]
        r_list = [int(x) for x in sorted(R)]
        result.append([l_list, r_list])
        
    return result

# EVOLVE-BLOCK-END

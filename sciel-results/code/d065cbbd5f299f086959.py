import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
        
    all_edges = list(edge_set)
    if not all_edges:
        return []

    # Generate candidate bicliques
    candidates = []
    
    # 1. Star bicliques (baseline)
    for u in range(n):
        if adj[u]:
            candidates.append((frozenset([u]), frozenset(adj[u])))
            
    # 2. Maximal bicliques containing each edge
    # To avoid duplicates and keep pool manageable, we use a set of (L, R) frozensets
    unique_candidates = set()
    for L, R in candidates:
        unique_candidates.add((L, R))
        
    for u, v in all_edges:
        # Start with L={u,v}, R=N(u) intersect N(v)
        # Note: u and v are in each other's neighborhoods
        R_init = adj[u] & adj[v]
        L_init = frozenset([u, v])
        R_cur = frozenset(R_init)
        L_cur = L_init
        
        # Expand R: add w if connected to all in L
        changed = True
        while changed:
            changed = False
            # Candidates for R: neighbors of u or v not already in L or R
            # Actually, any vertex w not in L_cur such that w is connected to all in L_cur
            # Since L_cur contains u and v, w must be in N(u) and N(v)
            # So w must be in N(u) & N(v)
            potential_R = (adj[u] & adj[v]) - L_cur
            for w in potential_R:
                if w not in R_cur:
                    # Check if w is connected to all in L_cur
                    # We know w is connected to u and v.
                    # If L_cur has more elements, we need to check them.
                    valid = True
                    for l in L_cur:
                        if w not in adj[l]:
                            valid = False
                            break
                    if valid:
                        R_cur = R_cur | {w}
                        changed = True
                        # We added to R, so potential L might change?
                        # If we add to R, L candidates must be connected to new R.
                        # So we should restart L expansion or check L validity.
                        # For simplicity and speed, we do a few rounds of mutual expansion.
                        pass
                        
        # Expand L: add w if connected to all in R
        changed = True
        while changed:
            changed = False
            # Candidates for L: neighbors of all in R?
            # w must be connected to all r in R_cur.
            # Since R_cur contains v, w must be in N(v).
            # Since R_cur contains u, w must be in N(u).
            # So w is in N(u) & N(v) & (intersection of N(r) for r in R_cur)
            # Let's just iterate over vertices not in L_cur or R_cur
            # Actually, w must be in N(u) & N(v) because u, v in R_cur? No, u,v in L_cur.
            # Wait, L_cur = {u, v}. R_cur contains neighbors.
            # If we add w to L, w must be connected to all r in R_cur.
            potential_L = set()
            # w must be connected to all in R_cur.
            # Start with intersection of neighbors of all r in R_cur
            if R_cur:
                common_neigh = adj[list(R_cur)[0]].copy()
                for r in R_cur:
                    common_neigh &= adj[r]
                potential_L = common_neigh - L_cur
            else:
                potential_L = set(range(n)) - L_cur
            
            for w in potential_L:
                if w not in L_cur:
                    # Check if w is connected to all in R_cur (already ensured by common_neigh)
                    # But we also need to ensure we don't break the biclique property for future additions?
                    # No, just add it.
                    L_cur = L_cur | {w}
                    changed = True
                    # Adding to L restricts R. So we should prune R?
                    # Or just restart the whole process?
                    # For a heuristic, let's just do one pass of L expansion after R expansion.
                    break # Break to restart L loop with new L? No, let's just add one and restart loop.
                    
        # Prune R: remove any r not connected to all in L_cur
        # And Prune L: remove any l not connected to all in R_cur
        # This ensures validity.
        valid_R = set()
        for r in R_cur:
            if all(r in adj[l] for l in L_cur):
                valid_R.add(r)
        R_cur = frozenset(valid_R)
        
        valid_L = set()
        for l in L_cur:
            if all(l in adj[r] for r in R_cur):
                valid_L.add(l)
        L_cur = frozenset(valid_L)
        
        if L_cur and R_cur:
            unique_candidates.add((L_cur, R_cur))
            
    candidates = list(unique_candidates)
    
    if not candidates:
        # Fallback to stars
        return [[[u], list(adj[u])] for u in range(n) if adj[u]]

    # Greedy Set Cover
    best_solution = None
    best_count = float('inf')
    
    # To improve, we can run greedy multiple times with random tie-breaking or candidate ordering
    # But for n=30, one good greedy pass might be enough.
    # Let's do a few random restarts of the greedy process.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        remaining_edges = set(all_edges)
        chosen_bicliques = []
        
        # Shuffle candidates to introduce randomness in tie-breaking
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        while remaining_edges:
            if time.time() > deadline:
                break
                
            best_biclique = None
            max_covered = -1
            best_edge_set = set()
            
            for L, R in shuffled_candidates:
                # Calculate edges covered by this biclique
                # Edges are pairs (l, r) for l in L, r in R
                # We only care about edges in remaining_edges
                covered = 0
                # Optimization: Precompute edge sets for candidates?
                # For n=30, L*R is small.
                for l in L:
                    for r in R:
                        e = (min(l, r), max(l, r))
                        if e in remaining_edges:
                            covered += 1
                
                if covered > max_covered:
                    max_covered = covered
                    best_biclique = (L, R)
                    # Store the specific edges covered for removal?
                    # We can recompute or store. Storing is safer.
                    best_edge_set = set()
                    for l in L:
                        for r in R:
                            e = (min(l, r), max(l, r))
                            if e in remaining_edges:
                                best_edge_set.add(e)
            
            if best_biclique is None or max_covered == 0:
                # Should not happen if candidates cover all edges
                # Fallback: pick any remaining edge and make a star
                if remaining_edges:
                    u, v = next(iter(remaining_edges))
                    # Use star at u
                    L = frozenset([u])
                    R = frozenset(adj[u])
                    # Intersect R with remaining edges' other endpoints?
                    # Just use the star, it covers u's edges.
                    chosen_bicliques.append((L, R))
                    # Remove edges covered by this star
                    for r in R:
                        e = (min(u, r), max(u, r))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                else:
                    break
            else:
                chosen_bicliques.append(best_biclique)
                remaining_edges -= best_edge_set
        
        if not remaining_edges:
            count = len(chosen_bicliques)
            if count < best_count:
                best_count = count
                best_solution = chosen_bicliques
                if best_count == 1: # Theoretical minimum
                    break

    if best_solution is None:
        # Fallback to star decomposition if greedy failed
        return [[[u], list(adj[u])] for u in range(n) if adj[u]]
    
    # Format output
    result = []
    for L, R in best_solution:
        result.append([list(L), list(R)])
        
    return result

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Normalize edges to frozensets for easy lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add(frozenset((u, v)))
    
    remaining_edges = set(edge_set)
    
    # Precompute adjacency matrix for O(1) edge check
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    def expand_biclique(u, v, rem_edges):
        """
        Expands the edge (u,v) into a maximal biclique within the subgraph defined by rem_edges.
        Uses a greedy strategy to add vertices.
        """
        L = {u}
        R = {v}
        
        # We want to maximize |L| * |R| or just |L x R| intersection with rem_edges
        # Greedy expansion:
        # Candidates for L: vertices connected to all in R
        # Candidates for R: vertices connected to all in L
        
        changed = True
        while changed:
            changed = False
            
            # Try to add to L
            best_l_candidate = None
            best_l_score = -1
            for u_cand in range(n):
                if u_cand in L or u_cand in R:
                    continue
                # Check if u_cand is connected to all in R
                can_add = True
                for v_r in R:
                    if not adj[u_cand][v_r]:
                        can_add = False
                        break
                    if frozenset((u_cand, v_r)) not in rem_edges:
                        can_add = False
                        break
                if can_add:
                    # Score: number of edges this vertex adds to the biclique
                    # It adds |R| edges
                    score = len(R)
                    # Tie-breaker: degree in remaining graph
                    if score > best_l_score:
                        best_l_score = score
                        best_l_candidate = u_cand
            
            if best_l_candidate is not None:
                L.add(best_l_candidate)
                changed = True
                
            # Try to add to R
            best_r_candidate = None
            best_r_score = -1
            for v_cand in range(n):
                if v_cand in L or v_cand in R:
                    continue
                # Check if v_cand is connected to all in L
                can_add = True
                for u_l in L:
                    if not adj[u_l][v_cand]:
                        can_add = False
                        break
                    if frozenset((u_l, v_cand)) not in rem_edges:
                        can_add = False
                        break
                if can_add:
                    score = len(L)
                    if score > best_r_score:
                        best_r_score = score
                        best_r_candidate = v_cand
            
            if best_r_candidate is not None:
                R.add(best_r_candidate)
                changed = True
                
        return L, R

    bicliques = []
    
    # Set Cover Greedy Approach
    # We will generate a pool of candidates and pick the one that covers the most remaining edges.
    NUM_CANDIDATES = 10
    
    while remaining_edges:
        candidates = []
        
        # 1. Find the edge with max degree sum to create a "strong" candidate
        degrees = {i: 0 for i in range(n)}
        for e in remaining_edges:
            u, v = tuple(e)
            degrees[u] += 1
            degrees[v] += 1
            
        best_edge = None
        best_score = -1
        for e in remaining_edges:
            u, v = tuple(e)
            score = degrees[u] + degrees[v]
            if score > best_score:
                best_score = score
                best_edge = e
        
        if best_edge:
            u, v = tuple(best_edge)
            L, R = expand_biclique(u, v, remaining_edges)
            # Calculate coverage
            covered = 0
            for lu in L:
                for rv in R:
                    if frozenset((lu, rv)) in remaining_edges:
                        covered += 1
            candidates.append((covered, L, R))
        
        # 2. Generate random candidates
        remaining_list = list(remaining_edges)
        for _ in range(NUM_CANDIDATES):
            if not remaining_list:
                break
            e = random.choice(remaining_list)
            u, v = tuple(e)
            L, R = expand_biclique(u, v, remaining_edges)
            covered = 0
            for lu in L:
                for rv in R:
                    if frozenset((lu, rv)) in remaining_edges:
                        covered += 1
            candidates.append((covered, L, R))
        
        # Pick the best candidate
        # Sort by coverage descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if not candidates:
            # Fallback: just take any edge as a 1x1 biclique
            e = next(iter(remaining_edges))
            u, v = tuple(e)
            L = {u}
            R = {v}
            bicliques.append([L, R])
            remaining_edges.remove(e)
            continue
        
        best_cov, best_L, best_R = candidates[0]
        
        if best_cov == 0:
             # Should not happen if remaining_edges is not empty and we picked from it
             # But if expansion failed?
             e = next(iter(remaining_edges))
             u, v = tuple(e)
             L = {u}
             R = {v}
             bicliques.append([L, R])
             remaining_edges.remove(e)
             continue

        # Remove covered edges
        for lu in best_L:
            for rv in best_R:
                rem_edge = frozenset((lu, rv))
                if rem_edge in remaining_edges:
                    remaining_edges.remove(rem_edge)
        
        bicliques.append([best_L, best_R])
        
    # Convert to lists of ints
    result = []
    for L, R in bicliques:
        L_list = [int(x) for x in sorted(L)]
        R_list = [int(x) for x in sorted(R)]
        result.append([L_list, R_list])
        
    return result
# EVOLVE-BLOCK-END

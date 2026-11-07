import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    n = P['n']
    edges = P['edges']
    
    # Build edge index map
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i
        
    m = len(edges)
    if m == 0:
        return []
        
    # Adjacency lists
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Bitmask adjacency
    # adj_mask[i] is a bitmask of vertices connected to i
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask

    # Enumerate Maximal Bicliques
    # A biclique is defined by (L, R) where L and R are sets of vertices.
    # We want to find all maximal bicliques.
    # Strategy:
    # For each vertex u, consider bicliques where u is the "pivot" (e.g., smallest vertex in L).
    # This avoids duplicates.
    
    bicliques = set() # Store edge masks
    
    # Helper to get edge mask for a biclique (L_mask, R_mask)
    def get_edge_mask(L_mask, R_mask):
        mask = 0
        # Iterate through vertices in L
        l_list = [i for i in range(n) if (L_mask >> i) & 1]
        for u in l_list:
            # Intersect neighbors of u with R
            common = adj_mask[u] & R_mask
            # The edges are (u, v) for v in common
            # We need to set bits in the edge mask
            for v in range(n):
                if (common >> v) & 1:
                    # Find edge index for (u, v)
                    if u < v:
                        idx = edge_index.get((u, v))
                    else:
                        idx = edge_index.get((v, u))
                    if idx is not None:
                        mask |= (1 << idx)
        return mask

    # Recursive function to expand bicliques
    # L_mask, R_mask: current biclique
    # X_mask: vertices that cannot be added to L (already tried or invalid)
    # Y_mask: vertices that cannot be added to R
    # u: the pivot vertex (smallest in L)
    
    def expand_biclique(L_mask, R_mask, X_mask, Y_mask, u):
        # Check for time
        if time.time() > deadline:
            return
        
        # Record this biclique if it has edges
        if L_mask and R_mask:
            emask = get_edge_mask(L_mask, R_mask)
            if emask:
                bicliques.add(emask)
        
        # Candidate vertices to add to L:
        # Must be connected to all in R.
        # And not in X_mask.
        # And > u to maintain ordering (u is smallest in L).
        
        # Common neighbors of all vertices in R
        # Start with all vertices, intersect with neighbors of each v in R
        common_L = (1 << n) - 1
        r_list = [i for i in range(n) if (R_mask >> i) & 1]
        for v in r_list:
            common_L &= adj_mask[v]
            
        # Candidates for L: common_L & ~X_mask & ~L_mask
        # We only add vertices > u to ensure u is the minimum in L
        # Actually, if u is the minimum, we shouldn't add any vertex < u.
        # So candidates must be > u.
        cand_L = common_L & ~X_mask & ~L_mask
        # Mask out vertices <= u
        mask_le_u = (1 << (u + 1)) - 1
        cand_L &= ~mask_le_u
        
        # Iterate through candidates for L
        while cand_L:
            # Pick a candidate
            l_bit = cand_L & -cand_L
            l_vertex = l_bit.bit_length() - 1
            cand_L ^= l_bit
            
            # New L
            new_L = L_mask | l_bit
            # New R: must be connected to all in new_L
            new_R = R_mask & adj_mask[l_vertex]
            
            # Recurse
            # X for L: X_mask | l_bit
            # Y for R: Y_mask (vertices not in new_R that were in R? No, standard BK logic)
            # In bipartite BK, we track exclusions.
            # If we add l_vertex to L, any vertex in R that is NOT connected to l_vertex cannot be in future R sets for this branch.
            # So new_Y = Y_mask | (R_mask & ~adj_mask[l_vertex])
            new_Y = Y_mask | (R_mask & ~adj_mask[l_vertex])
            
            expand_biclique(new_L, new_R, X_mask | l_bit, new_Y, u)
            
            # Also, we need to consider adding vertices to R?
            # The standard algorithm expands both sides.
            # To avoid duplicates, we fix the "root" u in L.
            # We expand L first. What about R?
            # If we only expand L, we might miss bicliques where we add to R first?
            # No, if we start with u in L, and expand L, we cover all bicliques containing u where u is min in L.
            # But what if we need to add a vertex to R that is not connected to the *current* L but is connected to a *subset*?
            # No, R must be a common neighborhood of L.
            # If we add to L, R shrinks.
            # What if we should have added a vertex to R *before* adding to L?
            # The initial R is N(u).
            # Any biclique containing u has R subset of N(u).
            # If we expand L, we are refining R.
            # This covers all bicliques where u is the min vertex in L.
            # Is that sufficient?
            # Yes, every biclique has a unique minimum vertex. If we assign that vertex to L, we find it.
            # Wait, the minimum vertex of the *union* L union R might be in R.
            # If min(L union R) is in R, then we won't find it when iterating u in L.
            # So we must ensure that for every biclique, the minimum vertex is in L.
            # How? We can just say: "The minimum vertex of the biclique must be in L".
            # This partitions the set of all bicliques.
            
            # Now, do we need to expand R?
            # If we fix u in L, and we expand L, we cover all bicliques containing u.
            # But what if we want to add a vertex to R?
            # R is determined by L. R = intersection of neighbors of L.
            # So we don't "choose" R vertices independently, they are forced by L.
            # So expanding L is sufficient to enumerate all bicliques containing u?
            # Yes.
            
        # Wait, the above logic only expands L.
        # Is it possible that a maximal biclique requires adding a vertex to R that is NOT in the common neighborhood of the current L?
        # No, by definition of biclique, R must be subset of common neighbors of L.
        # So R is always a subset of the initial R (which is N(u)).
        # And as we add to L, R shrinks.
        # So we are enumerating all bicliques containing u.
        
        # But what about the "Y" mask?
        # In standard BK, we also expand the other side.
        # Here, since R is derived from L, we don't need to branch on R additions.
        # We only branch on L additions.
        
    # Iterate over all vertices as potential minimum in L
    for u in range(n):
        if time.time() > deadline:
            break
        # Initial L: {u}
        # Initial R: N(u)
        # X: 0
        # Y: 0
        expand_biclique(1 << u, adj_mask[u], 0, 0, u)
        
    # If bicliques list is empty, fallback
    if not bicliques:
        # Use seed strategy
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
        
    # Convert bicliques to list
    biclique_list = list(bicliques)
    
    # Precompute for each edge, which bicliques cover it
    edge_to_bicliques = [[] for _ in range(m)]
    for b_idx, b_mask in enumerate(biclique_list):
        # Find bits set in b_mask
        temp = b_mask
        while temp:
            bit = temp & -temp
            e_idx = bit.bit_length() - 1
            edge_to_bicliques[e_idx].append(b_idx)
            temp ^= bit
            
    # Exact Cover Search
    best_count = len([u for u in range(n) if any(a==u for a,v in P['edges'])]) # Seed count
    best_solution = None
    current_solution = []
    
    # Sort bicliques by size descending to prefer large ones?
    # Or just use MRV in search.
    
    def search(remaining_mask, depth):
        nonlocal best_count, best_solution
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if depth < best_count:
                best_count = depth
                best_solution = current_solution[:]
            return
            
        if depth >= best_count:
            return
            
        # Find edge with fewest candidates
        min_count = m + 1
        min_edge = -1
        temp = remaining_mask
        while temp:
            bit = temp & -temp
            e_idx = bit.bit_length() - 1
            # Count valid bicliques for this edge
            # A biclique is valid if (b_mask & ~remaining_mask) == 0
            valid_count = 0
            for b_idx in edge_to_bicliques[e_idx]:
                if (biclique_list[b_idx] & ~remaining_mask) == 0:
                    valid_count += 1
            if valid_count < min_count:
                min_count = valid_count
                min_edge = e_idx
                if min_count == 1:
                    break
            temp ^= bit
            
        if min_edge == -1:
            return # No edges left? Should be handled by remaining_mask == 0
            
        if min_count == 0:
            return # Dead end
            
        # Try all valid bicliques for min_edge
        candidates = []
        for b_idx in edge_to_bicliques[min_edge]:
            b_mask = biclique_list[b_idx]
            if (b_mask & ~remaining_mask) == 0:
                candidates.append(b_idx)
                
        # Sort candidates by size descending (greedy heuristic)
        candidates.sort(key=lambda x: bin(biclique_list[x]).count('1'), reverse=True)
        
        for b_idx in candidates:
            b_mask = biclique_list[b_idx]
            current_solution.append(b_idx)
            search(remaining_mask ^ b_mask, depth + 1)
            current_solution.pop()
            
    search((1 << m) - 1, 0)
    
    # If no improvement found, return seed
    if best_solution is None:
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
        
    # Reconstruct bicliques from indices
    # We need to return list of [left, right]
    # We stored edge masks. We need to recover L and R for each edge mask.
    # This is hard. We should have stored L and R during enumeration.
    
    # Let's modify the enumeration to store (L_mask, R_mask, edge_mask)
    
    # Re-do enumeration with storage
    bicliques_data = [] # List of (L_mask, R_mask, edge_mask)
    seen_edge_masks = set()
    
    def expand_biclique_v2(L_mask, R_mask, X_mask, Y_mask, u):
        if time.time() > deadline:
            return
        
        if L_mask and R_mask:
            emask = get_edge_mask(L_mask, R_mask)
            if emask and emask not in seen_edge_masks:
                seen_edge_masks.add(emask)
                bicliques_data.append((L_mask, R_mask, emask))
        
        common_L = (1 << n) - 1
        r_list = [i for i in range(n) if (R_mask >> i) & 1]
        for v in r_list:
            common_L &= adj_mask[v]
            
        cand_L = common_L & ~X_mask & ~L_mask
        mask_le_u = (1 << (u + 1)) - 1
        cand_L &= ~mask_le_u
        
        while cand_L:
            l_bit = cand_L & -cand_L
            l_vertex = l_bit.bit_length() - 1
            cand_L ^= l_bit
            
            new_L = L_mask | l_bit
            new_R = R_mask & adj_mask[l_vertex]
            new_Y = Y_mask | (R_mask & ~adj_mask[l_vertex])
            
            expand_biclique_v2(new_L, new_R, X_mask | l_bit, new_Y, u)

    for u in range(n):
        if time.time() > deadline:
            break
        expand_biclique_v2(1 << u, adj_mask[u], 0, 0, u)
        
    if not bicliques_data:
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
        
    # Re-run search with new data
    biclique_list = [x[2] for x in bicliques_data]
    edge_to_bicliques = [[] for _ in range(m)]
    for b_idx, (L, R, emask) in enumerate(bicliques_data):
        temp = emask
        while temp:
            bit = temp & -temp
            e_idx = bit.bit_length() - 1
            edge_to_bicliques[e_idx].append(b_idx)
            temp ^= bit
            
    best_count = len([u for u in range(n) if any(a==u for a,v in P['edges'])])
    best_solution = None
    current_solution = []
    
    def search_v2(remaining_mask, depth):
        nonlocal best_count, best_solution
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if depth < best_count:
                best_count = depth
                best_solution = current_solution[:]
            return
            
        if depth >= best_count:
            return
            
        min_count = m + 1
        min_edge = -1
        temp = remaining_mask
        while temp:
            bit = temp & -temp
            e_idx = bit.bit_length() - 1
            valid_count = 0
            for b_idx in edge_to_bicliques[e_idx]:
                if (biclique_list[b_idx] & ~remaining_mask) == 0:
                    valid_count += 1
            if valid_count < min_count:
                min_count = valid_count
                min_edge = e_idx
                if min_count == 1:
                    break
            temp ^= bit
            
        if min_edge == -1:
            return
        if min_count == 0:
            return
            
        candidates = []
        for b_idx in edge_to_bicliques[min_edge]:
            b_mask = biclique_list[b_idx]
            if (b_mask & ~remaining_mask) == 0:
                candidates.append(b_idx)
                
        candidates.sort(key=lambda x: bin(biclique_list[x]).count('1'), reverse=True)
        
        for b_idx in candidates:
            b_mask = biclique_list[b_idx]
            current_solution.append(b_idx)
            search_v2(remaining_mask ^ b_mask, depth + 1)
            current_solution.pop()
            
    search_v2((1 << m) - 1, 0)
    
    if best_solution is None:
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
        
    result = []
    for b_idx in best_solution:
        L_mask, R_mask, _ = bicliques_data[b_idx]
        L = [i for i in range(n) if (L_mask >> i) & 1]
        R = [i for i in range(n) if (R_mask >> i) & 1]
        result.append([L, R])
        
    return result

# EVOLVE-BLOCK-END

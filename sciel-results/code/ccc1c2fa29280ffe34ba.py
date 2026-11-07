import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_masks = [0] * n
    edge_set = set()
    edge_index = {}
    edge_count = 0
    
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in edge_index:
            edge_index[(u, v)] = edge_count
            edge_count += 1
            edge_set.add((u, v))
        
    # Rebuild adj masks and edge_bit masks
    # We need adj_masks for biclique expansion
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # Precompute edge bit masks for pairs? 
    # Or just compute on the fly. 
    # Let's precompute a list of edges with their bit positions for faster mask creation.
    # Actually, for a biclique [L, R], we can compute the edge mask by:
    # mask = 0
    # for l in L:
    #    for r in R:
    #       if l < r: idx = edge_index.get((l,r))
    #       else: idx = edge_index.get((r,l))
    #       if idx is not None: mask |= (1 << idx)
    
    # To speed this up, we can precompute `edge_mask_contrib[v]`? No.
    # Given n=30, inner loops are small.
    
    candidates = []
    seen_bicliques = set()
    
    # Generate maximal bicliques for each edge (u, v)
    for u, v in edges:
        # Direction 1: u in L, v in R
        L = 1 << u
        R = 1 << v
        
        # Iterate to maximal
        changed = True
        while changed:
            changed = False
            # New L: intersection of neighbors of all in R
            new_L = (1 << n) - 1
            r_mask = R
            while r_mask:
                lsb = r_mask & -r_mask
                r_idx = lsb.bit_length() - 1
                new_L &= adj_masks[r_idx]
                r_mask ^= lsb
            
            # New R: intersection of neighbors of all in L
            new_R = (1 << n) - 1
            l_mask = L
            while l_mask:
                lsb = l_mask & -l_mask
                l_idx = lsb.bit_length() - 1
                new_R &= adj_masks[l_idx]
                l_mask ^= lsb
            
            if new_L != L:
                L = new_L
                changed = True
            if new_R != R:
                R = new_R
                changed = True
        
        # Check if u in L and v in R
        if (L & (1 << u)) and (R & (1 << v)):
            key = (L, R)
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                # Compute edge mask
                emask = 0
                l_mask = L
                while l_mask:
                    lsb = l_mask & -l_mask
                    l_idx = lsb.bit_length() - 1
                    r_mask = R
                    while r_mask:
                        rsb = r_mask & -r_mask
                        r_idx = rsb.bit_length() - 1
                        if l_idx < r_idx:
                            idx = edge_index.get((l_idx, r_idx))
                        else:
                            idx = edge_index.get((r_idx, l_idx))
                        if idx is not None:
                            emask |= (1 << idx)
                        r_mask ^= rsb
                    l_mask ^= lsb
                candidates.append((L, R, emask))
        
        # Direction 2: v in L, u in R
        L = 1 << v
        R = 1 << u
        changed = True
        while changed:
            changed = False
            new_L = (1 << n) - 1
            r_mask = R
            while r_mask:
                lsb = r_mask & -r_mask
                r_idx = lsb.bit_length() - 1
                new_L &= adj_masks[r_idx]
                r_mask ^= lsb
            
            new_R = (1 << n) - 1
            l_mask = L
            while l_mask:
                lsb = l_mask & -l_mask
                l_idx = lsb.bit_length() - 1
                new_R &= adj_masks[l_idx]
                l_mask ^= lsb
            
            if new_L != L:
                L = new_L
                changed = True
            if new_R != R:
                R = new_R
                changed = True
        
        if (L & (1 << v)) and (R & (1 << u)):
            key = (L, R)
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                emask = 0
                l_mask = L
                while l_mask:
                    lsb = l_mask & -l_mask
                    l_idx = lsb.bit_length() - 1
                    r_mask = R
                    while r_mask:
                        rsb = r_mask & -r_mask
                        r_idx = rsb.bit_length() - 1
                        if l_idx < r_idx:
                            idx = edge_index.get((l_idx, r_idx))
                        else:
                            idx = edge_index.get((r_idx, l_idx))
                        if idx is not None:
                            emask |= (1 << idx)
                        r_mask ^= rsb
                    l_mask ^= lsb
                candidates.append((L, R, emask))

    # Greedy Set Cover
    all_edges_mask = (1 << edge_count) - 1
    uncovered = all_edges_mask
    selected_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
        best_idx = -1
        best_count = -1
        for i, (L, R, emask) in enumerate(candidates):
            count = bin(emask & uncovered).count('1')
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        uncovered &= ~candidates[best_idx][2]
        
        if time.time() > deadline:
            break

    # Local Search
    # 1-swap: remove redundant, or replace with larger
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected_indices)):
            if time.time() > deadline:
                break
            # Calculate unique edges for selected[i]
            mask_without_i = 0
            for j in range(len(selected_indices)):
                if i != j:
                    mask_without_i |= candidates[selected_indices[j]][2]
            
            unique_edges = candidates[selected_indices[i]][2] & ~mask_without_i
            
            if unique_edges == 0:
                # Remove redundant
                selected_indices.pop(i)
                improved = True
                break # Restart loop
            
            # Try to replace with a candidate that covers unique_edges
            # and is "better" (e.g., covers more total edges, or just any valid one)
            # To keep it simple and fast, we look for a candidate that covers unique_edges
            # and has a larger total edge mask (to potentially cover more others)
            best_replacement = -1
            best_size = -1
            current_size = bin(candidates[selected_indices[i]][2]).count('1')
            
            for k, (L, R, emask) in enumerate(candidates):
                if k in selected_indices:
                    continue
                if (unique_edges & ~emask) == 0:
                    size = bin(emask).count('1')
                    if size > best_size:
                        best_size = size
                        best_replacement = k
            
            if best_replacement != -1:
                # Only replace if it seems beneficial? 
                # If we replace, we might enable removal of others.
                # Let's just replace if the new one is strictly larger in coverage
                if best_size > current_size:
                    selected_indices[i] = best_replacement
                    improved = True
                    break # Restart loop

    # Construct result
    result = []
    for idx in selected_indices:
        L, R, _ = candidates[idx]
        l_list = []
        l_mask = L
        while l_mask:
            lsb = l_mask & -l_mask
            l_list.append(lsb.bit_length() - 1)
            l_mask ^= lsb
        
        r_list = []
        r_mask = R
        while r_mask:
            rsb = r_mask & -r_mask
            r_list.append(rsb.bit_length() - 1)
            r_mask ^= rsb
        
        result.append([l_list, r_list])
        
    # If no bicliques found (should not happen), fallback to seed
    if not result:
         return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
         
    return result
# EVOLVE-BLOCK-END

import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Q3 adjacency in bitmask form for vertices 0..7
    # 0: 000, 1: 001, 2: 010, 3: 011, 4: 100, 5: 101, 6: 110, 7: 111
    q3_adj_masks = [0]*8
    q3_adj_masks[0] = (1<<1) | (1<<2) | (1<<4)
    q3_adj_masks[1] = (1<<0) | (1<<3) | (1<<5)
    q3_adj_masks[2] = (1<<0) | (1<<3) | (1<<6)
    q3_adj_masks[3] = (1<<1) | (1<<2) | (1<<7)
    q3_adj_masks[4] = (1<<0) | (1<<5) | (1<<6)
    q3_adj_masks[5] = (1<<1) | (1<<4) | (1<<7)
    q3_adj_masks[6] = (1<<2) | (1<<4) | (1<<7)
    q3_adj_masks[7] = (1<<3) | (1<<5) | (1<<6)

    def check_subset_q3(subset, adj_masks):
        # subset is a tuple of 8 vertex indices
        # adj_masks is a list of 16 integers
        # Check if induced subgraph on subset contains Q3
        
        # Map subset vertices to local indices 0..7
        # local_adj[i] is a bitmask of neighbors of local vertex i within the subset
        # We need to map global vertex indices to local indices.
        # local_index[global_vertex]
        
        # Quick check: Edge count
        # We can compute the adjacency matrix of the induced subgraph.
        # Let's just use the backtracking check directly on the global indices using adj_masks.
        
        # Degree check: all 8 vertices must have degree >= 3 in induced subgraph
        for i in range(8):
            deg = 0
            v = subset[i]
            for j in range(8):
                if i != j:
                    u = subset[j]
                    if (adj_masks[v] >> u) & 1:
                        deg += 1
            if deg < 3:
                return False

        # Backtracking search for isomorphism
        # Fix Q3 vertex 0 to subset vertex 0
        # Q3 neighbors of 0 are 1, 2, 4
        # Subset neighbors of subset[0]
        v0 = subset[0]
        neighbors_of_0 = []
        for j in range(8):
            if j != 0:
                vj = subset[j]
                if (adj_masks[v0] >> vj) & 1:
                    neighbors_of_0.append(j)
        
        if len(neighbors_of_0) < 3:
            return False
            
        # Try all combinations of 3 neighbors for Q3 vertices 1, 2, 4
        for combo in itertools.combinations(neighbors_of_0, 3):
            # combo[0] -> Q3 1, combo[1] -> Q3 2, combo[2] -> Q3 4
            # Remaining Q3 vertices: 3, 5, 6, 7
            # Remaining subset vertices: 8 - 1 (for 0) - 3 (for 1,2,4) = 4
            remaining_subset = [j for j in range(8) if j not in [0] and j not in combo]
            
            # Try permutations of remaining_subset for 3, 5, 6, 7
            for perm in itertools.permutations(remaining_subset):
                v3, v5, v6, v7 = perm
                
                # Check edges
                # 3 connected to 1, 2, 7
                if not ((adj_masks[subset[combo[0]]] >> subset[v3]) & 1 and 
                        (adj_masks[subset[combo[1]]] >> subset[v3]) & 1 and 
                        (adj_masks[subset[v3]] >> subset[v7]) & 1):
                    continue
                # 5 connected to 1, 4, 7
                if not ((adj_masks[subset[combo[0]]] >> subset[v5]) & 1 and 
                        (adj_masks[subset[combo[2]]] >> subset[v5]) & 1 and 
                        (adj_masks[subset[v5]] >> subset[v7]) & 1):
                    continue
                # 6 connected to 2, 4, 7
                if not ((adj_masks[subset[combo[1]]] >> subset[v6]) & 1 and 
                        (adj_masks[subset[combo[2]]] >> subset[v6]) & 1 and 
                        (adj_masks[subset[v6]] >> subset[v7]) & 1):
                    continue
                
                # If we reach here, all required edges are present.
                return True
                
        return False

    def find_one_q3(adj_masks):
        # Check all 8-vertex subsets
        # To speed up, we can randomize the order of subsets?
        # Or just iterate.
        for subset in itertools.combinations(range(16), 8):
            if check_subset_q3(subset, adj_masks):
                return subset
        return None

    def is_q3_free(adj_masks):
        return find_one_q3(adj_masks) is None

    # Precompute all subsets of size 8 containing a specific pair?
    # We can generate them on the fly or precompute a list of lists.
    # There are 120 pairs. Each pair is in binom(14,6)=3003 subsets.
    # Total entries: 120 * 3003 = 360360. This is small enough to precompute.
    
    pair_to_subsets = {}
    all_subsets = list(itertools.combinations(range(16), 8))
    for subset in all_subsets:
        for i in range(8):
            for j in range(i+1, 8):
                pair = (subset[i], subset[j])
                if pair not in pair_to_subsets:
                    pair_to_subsets[pair] = []
                pair_to_subsets[pair].append(subset)

    def check_pair_safe(u, v, adj_masks):
        # Check if adding edge (u,v) creates a Q3
        # Only need to check subsets containing u and v
        if (u, v) not in pair_to_subsets:
            return True # Should not happen for u != v
        for subset in pair_to_subsets[(u, v)]:
            if check_subset_q3(subset, adj_masks):
                return False
        return True

    best_adj_masks = [0]*16
    best_edges = 0
    
    # Try a few random starts
    for _ in range(5):
        if time.time() > deadline:
            break
            
        adj_masks = [0]*16
        for i in range(16):
            for j in range(i+1, 16):
                if random.random() < 0.6:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                    
        # Remove Q3s
        # While there is a Q3, remove an edge from one
        while time.time() < deadline:
            subset = find_one_q3(adj_masks)
            if subset is None:
                break
            
            # Pick a random edge in this subset to remove
            # We need to pick an edge that is part of the Q3?
            # Removing any edge from the induced subgraph might not break the Q3 if the Q3 uses a specific subgraph.
            # However, if we remove an edge (u,v) from the induced subgraph, and the Q3 used that edge, it's broken.
            # If the Q3 didn't use that edge, it might still exist.
            # So we should remove an edge that is *likely* part of the Q3.
            # Actually, if we remove a random edge from the induced subgraph, there's a chance it's not in the Q3.
            # Better: Find the Q3 mapping?
            # The current check_subset_q3 doesn't return the mapping.
            # Let's just remove a random edge from the subset. If it doesn't break the Q3, we'll find the Q3 again next iteration.
            # This is inefficient.
            
            # Improvement: Modify check_subset_q3 to return the edge to remove?
            # Or just remove all edges in the subset? No, that's too aggressive.
            
            # Let's try to identify an edge in the Q3.
            # We can modify check_subset_q3 to return a tuple (u,v) of an edge in the Q3 if found.
            
            # For now, let's just pick a random edge from the induced subgraph.
            edges_in_subset = []
            for i in range(8):
                for j in range(i+1, 8):
                    u = subset[i]
                    v = subset[j]
                    if (adj_masks[u] >> v) & 1:
                        edges_in_subset.append((u, v))
                        
            if edges_in_subset:
                u, v = random.choice(edges_in_subset)
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
            else:
                break # Should not happen if has_q3 is true

        # Now adj is Q3-free
        # Try to add edges
        # Iterate through all missing edges
        # If adding edge doesn't create Q3, add it
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(16):
                for j in range(i+1, 16):
                    if not ((adj_masks[i] >> j) & 1):
                        # Try adding
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        
                        # Check if safe
                        # We need to check if this addition created a Q3.
                        # We can check all subsets containing (i,j)
                        # But wait, the check_subset_q3 assumes the edge exists.
                        # So we can just call check_pair_safe(i, j, adj_masks)
                        # But check_pair_safe uses the current adj_masks which *includes* the new edge.
                        # So if check_subset_q3 finds a Q3, it means the new edge was necessary or part of it.
                        # Actually, check_subset_q3 checks if the induced subgraph contains Q3.
                        # If it returns True, then the new edge might have completed a Q3.
                        
                        if not check_pair_safe(i, j, adj_masks):
                            # Revert
                            adj_masks[i] &= ~(1 << j)
                            adj_masks[j] &= ~(1 << i)
                        else:
                            improved = True
                            
        edge_count = sum(bin(m).count('1') for m in adj_masks) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj_masks = adj_masks[:]
            
    # Convert best_adj_masks to list of edges
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if (best_adj_masks[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

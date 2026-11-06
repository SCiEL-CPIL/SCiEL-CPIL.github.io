import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    n = 16
    # Precompute all edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute Q3 subgraph masks for all 8-subsets of vertices
    # A subset is a tuple of 8 vertices
    q3_subsets = []
    for subset in combinations(range(n), 8):
        # Map subset vertices to 0..7
        # We need to check if the induced subgraph contains Q3
        # Q3 has 12 edges.
        # If the subset contains all 12 edges of some Q3 embedding, it's bad.
        # Actually, we just need to know if the induced subgraph *is* Q3 or contains it?
        # The task says "no subgraph isomorphic to Q3".
        # So if the 8 vertices induce a graph that contains Q3 as a subgraph, it's invalid.
        # Since Q3 has 12 edges, and the induced subgraph has at most 28 edges,
        # we can check if there is a bijection from Q3 vertices to these 8 vertices
        # such that all Q3 edges map to existing edges in the graph.
        
        # Optimization: 
        # Precompute for each 8-subset, the set of "Q3 patterns" (masks of edges in the complete graph K8)
        # that correspond to Q3.
        # Then, for a given graph, we check if any of these patterns is a subset of the graph's edges.
        
        # Let's precompute the masks of Q3 edges in K8.
        # There are many automorphisms. We can just list all 8! mappings? No, too many.
        # Better: Generate all subgraphs of K8 isomorphic to Q3.
        # There are 8! / 48 = 1344 distinct subgraphs? 
        # Actually, we can just iterate through all permutations of the 8 vertices?
        # 8! = 40320. For each permutation, check if it's a valid Q3 embedding?
        # No, we can just take the standard Q3 edges and permute the vertices.
        # But we only need the set of edge pairs in the local 0..7 indices.
        
        pass # This approach is getting complex. Let's stick to a faster backtracking or a smarter check.
        
    # Alternative Fast Check:
    # For a specific 8-subset, we can use the backtracking from the original code but optimized.
    # Or, we can precompute the "Q3 edge masks" for each 8-subset.
    
    # Let's try the precomputation of Q3 masks for each 8-subset.
    # For a fixed 8-subset S, let the vertices be v0..v7.
    # We want to know if there exists a bijection f: {0..7} -> S such that 
    # for all (u,v) in Q3_edges, (f(u), f(v)) is an edge in G.
    # This is equivalent to: Does the adjacency matrix of G[S] contain a subgraph isomorphic to Q3?
    
    # Let's precompute for each 8-subset S, the list of edge masks (as bitmasks over the 28 edges of K8)
    # that correspond to Q3 embeddings.
    # If the bitmask of G[S] (edges present) has a superset relationship with any Q3 mask, then Q3 is present.
    # i.e., if (G_mask & Q3_mask) == Q3_mask, then Q3 is present.
    
    # How many Q3 masks are there?
    # Number of subgraphs of K8 isomorphic to Q3.
    # We can generate them by taking the standard Q3 and permuting vertices 0..7.
    # But we must avoid duplicates.
    # Let's just generate all 40320 permutations and store the unique edge masks.
    
    q3_edges_local = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_local.add((i, j))
                
    # Map local edge (i,j) to bit index in 28-bit mask
    local_edge_to_bit = {}
    k8_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = len(k8_edges)
            k8_edges.append((i,j))
            
    q3_masks = set()
    # Generate all permutations of 0..7
    # This might be slow to precompute for all 128 subsets?
    # No, the set of Q3 masks in K8 is the same for all subsets!
    # Because the structure is isomorphic.
    # So we only need to compute the set of Q3 masks in K8 once.
    
    from itertools import permutations
    # 40320 permutations. For each, create a mask.
    # This takes a few seconds.
    perm_list = list(permutations(range(8)))
    for p in perm_list:
        mask = 0
        for u, v in q3_edges_local:
            # Map Q3 vertices u,v to permuted vertices p[u], p[v]
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    num_q3_masks = len(q3_masks_list)
    
    # Now, for each 8-subset of the 16 vertices, we need to map the global edges to the local 28 edges.
    # subset -> list of (global_edge_index, local_edge_bit)
    subset_edge_maps = {}
    for subset in combinations(range(n), 8):
        # Map local indices 0..7 to global indices subset[0]..subset[7]
        # For each local edge (i,j), find the global edge (subset[i], subset[j])
        # and get its index in all_edges.
        mapping = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                mapping.append((ge_idx, le_bit))
        subset_edge_maps[subset] = mapping
        
    def is_q3_free(graph_mask):
        # graph_mask is a bitmask of num_edges bits
        for subset, mapping in subset_edge_maps.items():
            # Build the local 28-bit mask for this subset
            local_mask = 0
            for ge_idx, le_bit in mapping:
                if (graph_mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            
            # Check if local_mask contains any Q3 mask
            for qm in q3_masks_list:
                if (local_mask & qm) == qm:
                    return False
        return True
        
    # Initial best: K3,13
    # Vertices 0,1,2 connected to 3..15
    best_mask = 0
    for u in range(3):
        for v in range(3, 16):
            idx = edge_index[(u,v)]
            best_mask |= (1 << idx)
            
    best_count = bin(best_mask).count('1')
    
    random.seed(42)
    
    # Local search with random restarts
    for trial in range(1000):
        if time.time() > deadline - 2:
            break
            
        # Random dense graph
        p = random.uniform(0.5, 0.8)
        current_mask = 0
        for i in range(num_edges):
            if random.random() < p:
                current_mask |= (1 << i)
                
        if bin(current_mask).count('1') <= best_count:
            continue
            
        # Greedy addition / removal to satisfy constraints and maximize edges
        # We want to add edges. If adding creates Q3, we might need to remove some.
        # Strategy: Try to add all missing edges one by one.
        # If it violates, skip it? Or try to remove a conflicting edge?
        # Simple greedy: Iterate through missing edges, try to add.
        # If Q3 appears, don't add.
        # This might get stuck in a local optimum.
        
        # Better: 
        # 1. Start with random mask.
        # 2. While Q3 exists:
        #      Find a Q3.
        #      Remove a random edge from it.
        # 3. While no Q3:
        #      Try to add a random missing edge.
        #      If Q3 created, remove it (or the added edge).
        
        # Let's implement the "repair then improve" strategy.
        
        # Step 1: Repair
        # Check if current_mask is Q3-free.
        # If not, find a violating Q3 and remove an edge.
        
        def find_violating_q3(mask):
            for subset, mapping in subset_edge_maps.items():
                local_mask = 0
                for ge_idx, le_bit in mapping:
                    if (mask >> ge_idx) & 1:
                        local_mask |= (1 << le_bit)
                for qm in q3_masks_list:
                    if (local_mask & qm) == qm:
                        # Found one. Return the global edge indices involved in this Q3 mask
                        # We need to map the bits in qm back to global indices
                        edges_to_remove = []
                        for i in range(8):
                            for j in range(i+1, 8):
                                le_bit = local_edge_to_bit[(i,j)]
                                if (qm >> le_bit) & 1:
                                    # Find global index for this local edge in this subset
                                    # We have mapping list. Find the pair (ge_idx, le_bit)
                                    # This is O(28) lookup.
                                    for ge_idx, l_bit in mapping:
                                        if l_bit == le_bit:
                                            edges_to_remove.append(ge_idx)
                                            break
                        return edges_to_remove
            return None
            
        # Repair loop
        max_repair_steps = 100
        for _ in range(max_repair_steps):
            viol_edges = find_violating_q3(current_mask)
            if viol_edges is None:
                break
            # Remove a random edge from the violation
            rem_idx = random.choice(viol_edges)
            current_mask &= ~(1 << rem_idx)
            
        if find_violating_q3(current_mask) is not None:
            # Failed to repair quickly, skip this trial
            continue
            
        # Improve loop
        # Try to add edges
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        random.shuffle(missing_edges)
        
        improved = True
        while improved:
            improved = False
            random.shuffle(missing_edges)
            for idx in missing_edges:
                if time.time() > deadline - 2:
                    break
                test_mask = current_mask | (1 << idx)
                if is_q3_free(test_mask):
                    current_mask = test_mask
                    missing_edges.remove(idx)
                    improved = True
                    if bin(current_mask).count('1') > best_count:
                        best_count = bin(current_mask).count('1')
                        best_mask = current_mask
            
        # Random flip / perturbation to escape local optima?
        # Maybe just restart is enough.
        
    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result

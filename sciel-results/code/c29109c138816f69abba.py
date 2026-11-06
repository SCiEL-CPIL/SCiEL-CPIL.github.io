import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    n = 16
    # Precompute all edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute Q3 subgraph masks for K8
    # Q3 vertices are 0..7. Edges connect i, j if popcount(i^j)==1.
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
            
    # Generate all unique Q3 masks in K8
    # There are 8! permutations, but many are symmetric.
    # We just generate all permutations and store the resulting bitmasks in a set.
    # 40320 permutations is manageable.
    q3_masks = set()
    # Optimization: We can fix vertex 0 to be mapped to 0,1,2,3 to reduce symmetry?
    # No, let's just do all permutations. It's a one-time cost.
    # Actually, we can be smarter. The automorphism group of Q3 has size 48.
    # So there are 8! / 48 = 1344 distinct embeddings.
    # Generating all 40320 and putting in a set is fine.
    
    # To speed up precomputation, we can iterate permutations
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    # Precompute mapping for each 8-subset to local edge bits
    # subset -> list of (global_edge_index, local_edge_bit)
    subsets = list(combinations(range(n), 8))
    subset_edge_maps = {}
    for subset in subsets:
        mapping = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                mapping.append((ge_idx, le_bit))
        subset_edge_maps[subset] = mapping
        
    def find_violating_q3(mask):
        """
        Returns a list of global edge indices forming a Q3, or None if Q3-free.
        """
        for subset, mapping in subset_edge_maps.items():
            # Build local 28-bit mask
            local_mask = 0
            for ge_idx, le_bit in mapping:
                if (mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            
            # Check against all Q3 masks
            for qm in q3_masks_list:
                if (local_mask & qm) == qm:
                    # Found a violation. Return the global edges involved.
                    # We need to map the bits in qm back to global indices.
                    viol_edges = []
                    # qm has 12 bits set.
                    # We can iterate through the mapping to find which global edges correspond to the set bits in qm.
                    for ge_idx, le_bit in mapping:
                        if (qm >> le_bit) & 1:
                            viol_edges.append(ge_idx)
                    return viol_edges
        return None

    def is_q3_free(mask):
        return find_violating_q3(mask) is None

    # Initial best: Empty graph or a known good start
    # Let's start with a random graph and repair it to get a valid start
    random.seed(42)
    
    best_mask = 0
    best_count = 0
    
    # Try to find a good initial solution quickly
    # Start with K3,13 (39 edges) as a safe baseline
    for u in range(3):
        for v in range(3, 16):
            idx = edge_index[(u,v)]
            best_mask |= (1 << idx)
    best_count = bin(best_mask).count('1')
    
    # Local search loop
    # We perform multiple restarts.
    for trial in range(200):
        if time.time() > deadline - 2:
            break
            
        # 1. Generate random initial mask
        p = random.uniform(0.6, 0.8)
        current_mask = 0
        for i in range(num_edges):
            if random.random() < p:
                current_mask |= (1 << i)
                
        # 2. Repair phase: Remove edges until Q3-free
        # Limit repair steps to avoid getting stuck
        for _ in range(200):
            viol_edges = find_violating_q3(current_mask)
            if viol_edges is None:
                break
            # Remove a random edge from the violation
            rem_idx = random.choice(viol_edges)
            current_mask &= ~(1 << rem_idx)
            
        # Check if repair was successful
        if find_violating_q3(current_mask) is not None:
            continue
            
        # 3. Improvement phase: Add edges
        # We try to add all missing edges.
        # To be efficient, we maintain a list of missing edges.
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        
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
            
        # 4. Perturbation phase
        # If we have time, try to escape local optima by removing some edges
        if time.time() > deadline - 5:
            continue
            
        # Remove a small number of random edges from the current best found in this trial
        # Or from the global best? Let's perturb the current state.
        current_edges = [i for i in range(num_edges) if (current_mask >> i) & 1]
        num_remove = random.randint(3, 8)
        if len(current_edges) > num_remove:
            to_remove = random.sample(current_edges, num_remove)
            for idx in to_remove:
                current_mask &= ~(1 << idx)
                
        # Re-improve
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
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

    # Convert best_mask to list of edges
    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result

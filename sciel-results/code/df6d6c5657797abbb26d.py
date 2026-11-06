import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    n = 16
    all_edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i, j)] = len(all_edges)
            all_edges.append((i, j))
    
    num_edges = len(all_edges)
    
    # Precompute Q3 masks in K8
    q3_edges_local = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_local.add((i, j))
                
    local_edge_to_bit = {}
    k8_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = len(k8_edges)
            k8_edges.append((i,j))
            
    q3_masks = set()
    from itertools import permutations
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
    q3_masks_list = list(q3_masks)
    
    # Precompute 8-subsets and their mappings
    subsets = list(combinations(range(n), 8))
    subset_mappings = []
    edge_to_subsets = {i: [] for i in range(num_edges)}
    
    for idx, subset in enumerate(subsets):
        mapping = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u, v)]
                le_bit = local_edge_to_bit[(i, j)]
                mapping.append((ge_idx, le_bit))
                edge_to_subsets[ge_idx].append(idx)
        subset_mappings.append(mapping)
        
    # Cache for contains_q3
    contains_q3_cache = {}
    
    def contains_q3(local_mask):
        if local_mask in contains_q3_cache:
            return contains_q3_cache[local_mask]
        
        for qm in q3_masks_list:
            if (local_mask & qm) == qm:
                contains_q3_cache[local_mask] = True
                return True
        contains_q3_cache[local_mask] = False
        return False
        
    def find_violating_q3_in_subset(s_idx, mask):
        mapping = subset_mappings[s_idx]
        local_mask = 0
        for ge_idx, le_bit in mapping:
            if (mask >> ge_idx) & 1:
                local_mask |= (1 << le_bit)
        for qm in q3_masks_list:
            if (local_mask & qm) == qm:
                # Return global edges of the violation
                viol_edges = []
                for ge_idx, le_bit in mapping:
                    if (qm >> le_bit) & 1:
                        viol_edges.append(ge_idx)
                return viol_edges
        return None

    def is_safe_to_add(edge_idx, mask):
        # Check all subsets containing this edge
        for s_idx in edge_to_subsets[edge_idx]:
            mapping = subset_mappings[s_idx]
            local_mask = 0
            for ge_idx, le_bit in mapping:
                if (mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            # Note: The new edge is not in 'mask' yet, so local_mask doesn't include it.
            # We need to check if adding it creates a Q3.
            # So we should check if (local_mask | (1 << bit_of_new_edge)) contains Q3?
            # Wait, the new edge IS in the subset.
            # We need to find the bit index of the new edge in this subset.
            new_edge_bit = None
            for ge_idx, le_bit in mapping:
                if ge_idx == edge_idx:
                    new_edge_bit = le_bit
                    break
            
            test_local_mask = local_mask | (1 << new_edge_bit)
            if contains_q3(test_local_mask):
                return False
        return True

    def find_any_violation(mask):
        # This is used for repair. We can pick a random subset?
        # Or iterate through all subsets?
        # Since we only add edges if safe, violations only appear if we start with a bad state or during perturbation.
        # For repair, we just need to find ONE violation.
        # We can check random subsets until we find one.
        # Or we can iterate through all 12870.
        # Let's iterate through a random sample first.
        random.shuffle(subsets_indices_cache) # Need a list of indices
        for s_idx in subsets_indices_cache:
            viol = find_violating_q3_in_subset(s_idx, mask)
            if viol:
                return viol
        # Fallback: full scan
        for s_idx in range(len(subsets)):
            viol = find_violating_q3_in_subset(s_idx, mask)
            if viol:
                return viol
        return None

    subsets_indices_cache = list(range(len(subsets)))
    
    # Initial best
    random.seed(42)
    best_mask = 0
    best_count = 0
    
    # Start with K3,13
    for u in range(3):
        for v in range(3, 16):
            idx = edge_index[(u, v)]
            best_mask |= (1 << idx)
    best_count = bin(best_mask).count('1')
    
    current_mask = best_mask
    
    # Local Search
    for trial in range(10000):
        if time.time() > deadline - 2:
            break
            
        # If current is not Q3-free, repair it
        viol = find_any_violation(current_mask)
        while viol:
            rem_idx = random.choice(viol)
            current_mask &= ~(1 << rem_idx)
            viol = find_any_violation(current_mask)
            if time.time() > deadline - 2:
                break
        
        if time.time() > deadline - 2:
            break
            
        # Improvement Phase
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        random.shuffle(missing_edges)
        
        improved = True
        while improved:
            improved = False
            for idx in missing_edges:
                if time.time() > deadline - 2:
                    break
                if is_safe_to_add(idx, current_mask):
                    current_mask |= (1 << idx)
                    missing_edges.remove(idx)
                    improved = True
                    if bin(current_mask).count('1') > best_count:
                        best_count = bin(current_mask).count('1')
                        best_mask = current_mask
                        
        # Perturbation
        if time.time() > deadline - 5:
            continue
            
        current_edges = [i for i in range(num_edges) if (current_mask >> i) & 1]
        if len(current_edges) > 5:
            num_remove = random.randint(3, 8)
            to_remove = random.sample(current_edges, num_remove)
            for idx in to_remove:
                current_mask &= ~(1 << idx)
                
    # Final Polish
    if time.time() < deadline - 2:
        missing_edges = [i for i in range(num_edges) if not (best_mask >> i) & 1]
        random.shuffle(missing_edges)
        for idx in missing_edges:
            if time.time() > deadline - 1:
                break
            if is_safe_to_add(idx, best_mask):
                best_mask |= (1 << idx)
                best_count = bin(best_mask).count('1')

    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result

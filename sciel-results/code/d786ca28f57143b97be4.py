import time
import random
from itertools import combinations, permutations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 16
    # All possible edges in K16
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Q3 structure: 8 vertices, 12 edges (Hamming distance 1)
    q3_edges_local = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_local.add((i, j))
    
    # Map local K8 edges to bit positions
    local_edge_to_bit = {}
    k8_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = len(k8_edges)
            k8_edges.append((i,j))
    
    # Precompute all Q3 masks in K8 (28-bit masks)
    # A mask represents which of the 28 edges of K8 are present in the Q3 subgraph
    q3_masks = set()
    # There are 8! = 40320 permutations of 8 vertices
    # But many are symmetric; we can generate all distinct masks
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b:
                a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
    
    q3_masks_list = list(q3_masks)
    
    # Precompute has_q3 lookup table for 28-bit masks
    # has_q3[mask] = 1 if mask contains any Q3 subgraph
    # This is a 2^28 array, which is 256MB as uint8 - might be too much
    # Instead, use a smarter approach: for each mask, check if it contains any q3_mask
    # But that's slow. Let's use a different strategy.
    
    # Alternative: Precompute for each 8-vertex subset of the 16-vertex graph,
    # the mapping from global edge indices to local K8 edge positions.
    
    # Precompute subset edge mappings
    # For each 8-subset of {0..15}, store the 28 global edge indices corresponding to K8 edges
    subset_edge_maps = {}
    for subset in combinations(range(n), 8):
        mapping = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v:
                    u, v = v, u
                ge_idx = edge_index[(u,v)]
                mapping.append(ge_idx)
        subset_edge_maps[subset] = mapping
    
    # For each global edge, which subsets contain it?
    edge_to_subsets = {i: [] for i in range(num_edges)}
    for subset, mapping in subset_edge_maps.items():
        for ge_idx in mapping:
            edge_to_subsets[ge_idx].append(subset)
    
    # Fast Q3 check: given a set of edges (as a boolean array or set),
    # check if any 8-vertex subset induces a Q3 subgraph.
    # We'll represent the graph as a set of edge indices.
    
    def extract_local_mask(edge_set, subset):
        """Extract the 28-bit local mask for a given 8-vertex subset."""
        mapping = subset_edge_maps[subset]
        local_mask = 0
        for k, ge_idx in enumerate(mapping):
            if ge_idx in edge_set:
                local_mask |= (1 << k)
        return local_mask
    
    def is_q3_free(edge_set, subsets_to_check=None):
        """Check if the graph defined by edge_set is Q3-free."""
        if subsets_to_check is None:
            subsets_to_check = list(subset_edge_maps.keys())
        
        for subset in subsets_to_check:
            lm = extract_local_mask(edge_set, subset)
            # Check if lm contains any Q3 mask
            for qm in q3_masks_list:
                if (lm & qm) == qm:
                    return False
        return True
    
    def find_violation(edge_set, subsets_to_check=None):
        """Find a Q3 subgraph if one exists. Returns set of edge indices forming Q3, or None."""
        if subsets_to_check is None:
            subsets_to_check = list(subset_edge_maps.keys())
        
        for subset in subsets_to_check:
            lm = extract_local_mask(edge_set, subset)
            for qm in q3_masks_list:
                if (lm & qm) == qm:
                    # This Q3 is present. Return the global edge indices.
                    mapping = subset_edge_maps[subset]
                    viol_edges = set()
                    for k in range(28):
                        if (qm >> k) & 1:
                            viol_edges.add(mapping[k])
                    return viol_edges
        return None
    
    # Initial best: K_{3,13} = 39 edges (known Q3-free)
    best_edge_set = set()
    for u in range(3):
        for v in range(3, 16):
            best_edge_set.add(edge_index[(u,v)])
    best_count = len(best_edge_set)
    
    random.seed(42)
    
    # Local search: start from random dense graphs, repair, then improve
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        # Start with a random graph with density ~0.5-0.7
        p = random.uniform(0.45, 0.65)
        current_set = set()
        for i in range(num_edges):
            if random.random() < p:
                current_set.add(i)
        
        # Repair phase: remove edges to eliminate Q3
        repair_iters = 0
        while time.time() < deadline - 15 and repair_iters < 100:
            viol = find_violation(current_set)
            if viol is None:
                break
            # Remove a random edge from the violation
            rem = random.choice(list(viol))
            current_set.remove(rem)
            repair_iters += 1
        
        if time.time() > deadline - 10:
            break
        
        # Check if repair succeeded
        if find_violation(current_set) is not None:
            continue
        
        # Improve phase: try to add edges
        missing_edges = [i for i in range(num_edges) if i not in current_set]
        random.shuffle(missing_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(missing_edges)
            for idx in missing_edges:
                if time.time() > deadline - 5:
                    break
                # Try adding this edge
                test_set = current_set | {idx}
                # Only need to check subsets containing this edge
                subs = edge_to_subsets[idx]
                if is_q3_free(test_set, subs):
                    current_set = test_set
                    missing_edges.remove(idx)
                    improved = True
                    if len(current_set) > best_count:
                        best_count = len(current_set)
                        best_edge_set = set(current_set)
        
        # Update best
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_edge_set = set(current_set)
    
    # Final verification
    if not is_q3_free(best_edge_set):
        # Fallback to K_{3,13}
        best_edge_set = set()
        for u in range(3):
            for v in range(3, 16):
                best_edge_set.add(edge_index[(u,v)])
    
    # Convert to list of [u, v] pairs
    result = []
    for idx in best_edge_set:
        u, v = all_edges[idx]
        result.append([u, v])
    
    return result

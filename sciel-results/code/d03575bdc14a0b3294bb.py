import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute Q3 edge masks in K8
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
    
    from itertools import permutations
    q3_masks = set()
    perm_list = list(permutations(range(8)))
    for p in perm_list:
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
    q3_masks_list = list(q3_masks)
    num_q3_masks = len(q3_masks_list)
    q3_masks_np = np.array(q3_masks_list, dtype=np.uint32)
    
    # Precompute for each 8-subset: mapping from local edge bit to global edge index
    subset_list = list(combinations(range(n), 8))
    num_subsets = len(subset_list)
    
    subset_data = []
    subset_edge_indices = []
    for subset in subset_list:
        local_to_global = [0] * 28
        indices = [0] * 28
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                local_to_global[le_bit] = ge_idx
                indices[le_bit] = ge_idx
        subset_data.append(local_to_global)
        subset_edge_indices.append(indices)

    def find_violating_q3(mask):
        """Find a Q3 in the graph. Returns list of global edge indices or None."""
        for si in range(num_subsets):
            indices = subset_edge_indices[si]
            local_mask = 0
            for le_bit in range(28):
                ge_idx = indices[le_bit]
                if (mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            
            if local_mask.bit_count() < 12:
                continue
                
            lm_np = np.uint32(local_mask)
            matches = (lm_np & q3_masks_np) == q3_masks_np
            if np.any(matches):
                idx = np.argmax(matches)
                qm = q3_masks_list[idx]
                edges = []
                ltg = subset_data[si]
                for le_bit in range(28):
                    if (qm >> le_bit) & 1:
                        edges.append(ltg[le_bit])
                return edges
        return None

    # Greedy initialization
    mask = (1 << num_edges) - 1
    edge_count = num_edges
    
    max_removals = 200
    for _ in range(max_removals):
        if time.time() > deadline - 30:
            break
        viol = find_violating_q3(mask)
        if viol is None:
            break
        rem_idx = random.choice(viol)
        mask &= ~(1 << rem_idx)
        edge_count -= 1
        
    best_mask = mask
    best_count = edge_count
    
    random.seed(42)
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.99
    
    trial = 0
    while time.time() < deadline - 5:
        trial += 1
        
        if trial % 100 == 0:
            # Perturb best
            mask = best_mask
            edge_count = best_count
            for _ in range(random.randint(5, 15)):
                present = [i for i in range(num_edges) if (mask >> i) & 1]
                if not present: break
                rem = random.choice(present)
                mask &= ~(1 << rem)
                edge_count -= 1
            # Repair
            for _ in range(50):
                viol = find_violating_q3(mask)
                if viol is None: break
                rem_idx = random.choice(viol)
                mask &= ~(1 << rem_idx)
                edge_count -= 1
            if find_violating_q3(mask) is not None:
                continue
                
        # Local Search Step: Try to add an edge
        missing = [i for i in range(num_edges) if not (mask >> i) & 1]
        if missing:
            idx = random.choice(missing)
            test_mask = mask | (1 << idx)
            if find_violating_q3(test_mask) is None:
                mask = test_mask
                edge_count += 1
                if edge_count > best_count:
                    best_count = edge_count
                    best_mask = mask
        
        # Swap: Remove one, Add one
        if random.random() < 0.5:
            present = [i for i in range(num_edges) if (mask >> i) & 1]
            missing = [i for i in range(num_edges) if not (mask >> i) & 1]
            if present and missing:
                rem = random.choice(present)
                add = random.choice(missing)
                test_mask = (mask & ~(1 << rem)) | (1 << add)
                if find_violating_q3(test_mask) is None:
                    mask = test_mask
        
        T = max(T_min, T * cooling)

    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 16
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Generate Q3 masks
    k8_edges = []
    local_edge_to_bit = {}
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = len(k8_edges)
            k8_edges.append((i,j))
            
    from itertools import permutations
    q3_local_masks = set()
    for p in permutations(range(8)):
        mask = 0
        for u in range(8):
            for v in range(u+1, 8):
                if bin(u ^ v).count('1') == 1:
                    a, b = p[u], p[v]
                    if a > b: a, b = b, a
                    bit = local_edge_to_bit[(a,b)]
                    mask |= (1 << bit)
        q3_local_masks.add(mask)
        
    q3_local_list = list(q3_local_masks)
    
    all_q3_masks = []
    q3s_with_edge = [[] for _ in range(num_edges)]
    
    subsets = list(combinations(range(n), 8))
    
    for subset in subsets:
        local_to_global_edge = [0]*28
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                local_to_global_edge[le_bit] = ge_idx
                
        for lm in q3_local_list:
            global_mask = 0
            q3_edges_global = []
            for le_bit in range(28):
                if (lm >> le_bit) & 1:
                    ge_idx = local_to_global_edge[le_bit]
                    global_mask |= (1 << ge_idx)
                    q3_edges_global.append(ge_idx)
            all_q3_masks.append((global_mask, q3_edges_global))
            
    num_q3s = len(all_q3_masks)
    
    for q_idx, (_, q3_edges_global) in enumerate(all_q3_masks):
        for e in q3_edges_global:
            q3s_with_edge[e].append(q_idx)

    best_count = 0
    best_mask = 0
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
            
        current_mask = 0
        q3_counts = [0] * num_q3s
        
        # Greedy Initialization
        edges_to_try = list(range(num_edges))
        random.shuffle(edges_to_try)
        for e in edges_to_try:
            can_add = True
            for q_idx in q3s_with_edge[e]:
                if q3_counts[q_idx] == 11:
                    can_add = False
                    break
            if can_add:
                current_mask |= (1 << e)
                for q_idx in q3s_with_edge[e]:
                    q3_counts[q_idx] += 1
                    
        # Local Search: Swap + Greedy Add
        max_swaps = 50
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
                
            # Greedy Add
            improved = True
            while improved:
                improved = False
                missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
                if not missing_edges:
                    break
                random.shuffle(missing_edges)
                for e in missing_edges:
                    can_add = True
                    for q_idx in q3s_with_edge[e]:
                        if q3_counts[q_idx] == 11:
                            can_add = False
                            break
                    if can_add:
                        current_mask |= (1 << e)
                        for q_idx in q3s_with_edge[e]:
                            q3_counts[q_idx] += 1
                        improved = True
                        break
                        
            if time.time() > deadline:
                break

            if not improved:
                present_edges = [i for i in range(num_edges) if (current_mask >> i) & 1]
                missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
                
                if not present_edges or not missing_edges:
                    break
                    
                e_out = random.choice(present_edges)
                e_in = random.choice(missing_edges)
                
                for q_idx in q3s_with_edge[e_out]:
                    q3_counts[q_idx] -= 1
                    
                can_add_in = True
                for q_idx in q3s_with_edge[e_in]:
                    if q3_counts[q_idx] == 11:
                        can_add_in = False
                        break
                        
                if can_add_in:
                    current_mask &= ~(1 << e_out)
                    current_mask |= (1 << e_in)
                    for q_idx in q3s_with_edge[e_in]:
                        q3_counts[q_idx] += 1
                else:
                    for q_idx in q3s_with_edge[e_out]:
                        q3_counts[q_idx] += 1

        count = bin(current_mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = current_mask
            
    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result

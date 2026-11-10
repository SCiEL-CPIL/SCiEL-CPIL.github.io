import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    
    # Precomputation
    # 1. Edge indices
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
    total_edges = 120
    
    # 2. Subsets of size 8
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)
    
    # 3. Local edge indices for K8
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
            
    # 4. Generate Q3 masks for K8
    q3_masks_for_k8 = []
    seen_masks = set()
    for A in itertools.combinations(range(8), 4):
        B = tuple(v for v in range(8) if v not in A)
        for perm in itertools.permutations(B):
            mask = 0
            for i in range(4):
                a = A[i]
                m = perm[i]
                for j in range(4):
                    b = B[j]
                    if b != m:
                        e_idx = local_edge_idx[a][b]
                        mask |= (1 << e_idx)
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks_for_k8.append(mask)
                
    # 5. Map global edges to subsets and local indices
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1]*total_edges for _ in range(num_subsets)]
    
    # Also precompute reverse mapping: for each subset, local_pos -> global_edge
    # This is useful for swap operations
    local_to_global = [{} for _ in range(num_subsets)]
    
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                l_pos = local_edge_idx[i][j]
                
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos
                local_to_global[s_idx][l_pos] = g_idx

    def contains_q3(mask):
        if mask.bit_count() < 12:
            return False
        for q in q3_masks_for_k8:
            if (mask & q) == q:
                return True
        return False

    def can_add_edge(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            if contains_q3(new_mask):
                return False
        return True

    def is_valid_swap(G, e_add, e_rem, subset_masks):
        """Check if G - e_rem + e_add is Q3-free.
        Since G is Q3-free, any new Q3 must contain e_add.
        So we check: for all subsets containing e_add, does (subset_mask - e_rem + e_add) contain a Q3?
        """
        # For each subset containing e_add:
        for s_idx in edge_to_subsets[e_add]:
            l_pos_add = local_idx[s_idx][e_add]
            
            # Remove e_rem from this subset's mask if it's in this subset
            if e_rem in local_to_global[s_idx].values():
                # Find local pos of e_rem in this subset
                l_pos_rem = local_idx[s_idx][e_rem]
                new_mask = (subset_masks[s_idx] & ~(1 << l_pos_rem)) | (1 << l_pos_add)
            else:
                new_mask = subset_masks[s_idx] | (1 << l_pos_add)
            
            if contains_q3(new_mask):
                return False
        
        # Also need to check: does removing e_rem and adding e_add create issues in subsets NOT containing e_add?
        # No, because e_add is only in subsets containing it.
        # But we need to make sure that removing e_rem doesn't leave some subset in a bad state?
        # No, removing edges can't create new Q3s.
        return True

    best_G = 0
    best_count = 0
    
    # Multiple restarts
    while time.time() < deadline:
        G = 0
        subset_masks = [0] * num_subsets
        missing_edges = set(range(total_edges))
        
        # Greedy Phase
        edges = list(range(total_edges))
        random.shuffle(edges)
        
        for e_idx in edges:
            if time.time() > deadline:
                break
            if can_add_edge(G, e_idx, subset_masks):
                G |= (1 << e_idx)
                missing_edges.discard(e_idx)
                for s_idx in edge_to_subsets[e_idx]:
                    l_pos = local_idx[s_idx][e_idx]
                    subset_masks[s_idx] |= (1 << l_pos)
        
        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G
        
        # Local Search / Simulated Annealing Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing edge
            missing_list = list(missing_edges)
            random.shuffle(missing_list)
            
            for e_idx in missing_list:
                if can_add_edge(G, e_idx, subset_masks):
                    G |= (1 << e_idx)
                    missing_edges.discard(e_idx)
                    for s_idx in edge_to_subsets[e_idx]:
                        l_pos = local_idx[s_idx][e_idx]
                        subset_masks[s_idx] |= (1 << l_pos)
                    improved = True
                    count = G.bit_count()
                    if count > best_count:
                        best_count = count
                        best_G = G
                    break
            
            # 2. If no add, try a swap (remove one edge, add another)
            if not improved and missing_edges:
                # Pick a random missing edge to try to add
                e_add = random.choice(list(missing_edges))
                
                # Find which subsets block this edge
                blocking_subsets = []
                for s_idx in edge_to_subsets[e_add]:
                    l_pos_add = local_idx[s_idx][e_add]
                    new_mask = subset_masks[s_idx] | (1 << l_pos_add)
                    if contains_q3(new_mask):
                        blocking_subsets.append(s_idx)
                
                if blocking_subsets:
                    # Pick a random blocking subset
                    s_idx = random.choice(blocking_subsets)
                    
                    # Find candidate edges to remove from this subset that are in G
                    s_vertices = subsets_8[s_idx]
                    candidate_removals = []
                    for i in range(8):
                        for j in range(i+1, 8):
                            u_glob = s_vertices[i]
                            v_glob = s_vertices[j]
                            g_idx = edge_index[u_glob][v_glob]
                            if (G >> g_idx) & 1 and g_idx != e_add:
                                candidate_removals.append(g_idx)
                    
                    random.shuffle(candidate_removals)
                    
                    for e_rem in candidate_removals:
                        if is_valid_swap(G, e_add, e_rem, subset_masks):
                            # Perform the swap
                            # Remove e_rem
                            G &= ~(1 << e_rem)
                            for s_idx2 in edge_to_subsets[e_rem]:
                                l_pos = local_idx[s_idx2][e_rem]
                                subset_masks[s_idx2] &= ~(1 << l_pos)
                            missing_edges.add(e_rem)
                            
                            # Add e_add
                            G |= (1 << e_add)
                            missing_edges.discard(e_add)
                            for s_idx2 in edge_to_subsets[e_add]:
                                l_pos = local_idx[s_idx2][e_add]
                                subset_masks[s_idx2] |= (1 << l_pos)
                            
                            improved = True
                            count = G.bit_count()
                            if count > best_count:
                                best_count = count
                                best_G = G
                            break

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result

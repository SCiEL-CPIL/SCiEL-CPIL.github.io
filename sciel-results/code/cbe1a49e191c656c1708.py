import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    
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
    num_local_edges = 28
            
    # 4. Generate Q3 masks for K8
    # Canonical Q3 edges: (i, j) where i^j is power of 2
    q3_canonical_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            d = i ^ j
            if d & (d-1) == 0:
                q3_canonical_edges.append((i, j))
    
    q3_masks_for_k8 = []
    seen_masks = set()
    
    for perm in itertools.permutations(range(8)):
        mask = 0
        for (u, v) in q3_canonical_edges:
            pu = perm[u]
            pv = perm[v]
            e_idx = local_edge_idx[pu][pv]
            mask |= (1 << e_idx)
        if mask not in seen_masks:
            seen_masks.add(mask)
            q3_masks_for_k8.append(mask)
            
    # 5. Precompute "Rest Masks" for fast Q3 check
    # For each local edge b (0..27), store list of masks (Q \ {b}) for all Q3s Q containing b.
    # If current subset mask M contains (Q \ {b}), then adding b creates Q.
    rest_masks_for_local_edge = [[] for _ in range(num_local_edges)]
    for q_mask in q3_masks_for_k8:
        # Iterate through bits in q_mask
        m = q_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            b = lsb.bit_length() - 1
            rest = q_mask ^ lsb
            rest_masks_for_local_edge[b].append(rest)
            m ^= lsb
            
    # 6. Map global edges to subsets and local indices
    edge_to_subsets = [[] for _ in range(total_edges)]
    # local_idx[s_idx][g_idx] = local bit position
    local_idx = [[-1]*total_edges for _ in range(num_subsets)]
    
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                l_pos = local_edge_idx[i][j]
                
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos

    # Helper: Check if adding local edge b to subset mask M creates a Q3
    def is_blocked(M, b):
        for rest in rest_masks_for_local_edge[b]:
            if (M & rest) == rest:
                return True
        return False

    # Helper: Check if global edge e_idx can be added to G given subset_masks
    def can_add_edge(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            b = local_idx[s_idx][e_idx]
            if is_blocked(subset_masks[s_idx], b):
                return False
        return True

    # Greedy Construction with smart ordering
    def greedy_construct(order):
        G = 0
        subset_masks = [0] * num_subsets
        for e_idx in order:
            if can_add_edge(G, e_idx, subset_masks):
                G |= (1 << e_idx)
                for s_idx in edge_to_subsets[e_idx]:
                    b = local_idx[s_idx][e_idx]
                    subset_masks[s_idx] |= (1 << b)
        return G, subset_masks

    # Check if swap (remove e_rem, add e_add) is valid
    def is_valid_swap(e_rem, e_add, subset_masks):
        # The swap is valid if for ALL Q3s Q containing e_add:
        #   Either G does NOT contain Q \ {e_add}, OR e_rem is in Q \ {e_add}.
        for s_idx in edge_to_subsets[e_add]:
            b_add = local_idx[s_idx][e_add]
            b_rem = local_idx[s_idx][e_rem]
            for rest in rest_masks_for_local_edge[b_add]:
                if (subset_masks[s_idx] & rest) == rest:
                    # G contains Q \ {e_add}, so we need e_rem to be in rest
                    if b_rem == -1 or not ((rest >> b_rem) & 1):
                        return False
        return True

    # Try to find a valid add edge from a list
    def find_valid_add(missing_edges, subset_masks, max_tries=50):
        for _ in range(max_tries):
            if not missing_edges:
                return -1
            e_add = random.choice(missing_edges)
            if can_add_edge(0, e_add, subset_masks):
                return e_add
        return -1

    # Try to find a valid swap
    def find_valid_swap(missing_edges, present_edges, subset_masks, max_tries=30):
        for _ in range(max_tries):
            if not missing_edges or not present_edges:
                break
            e_add = random.choice(missing_edges)
            e_rem = random.choice(present_edges)
            if is_valid_swap(e_rem, e_add, subset_masks):
                return e_rem, e_add
        return -1, -1

    # Perform an add
    def do_add(e_add, current_G, subset_masks, missing_edges, present_edges):
        current_G |= (1 << e_add)
        missing_edges.remove(e_add)
        present_edges.append(e_add)
        for s_idx in edge_to_subsets[e_add]:
            b = local_idx[s_idx][e_add]
            subset_masks[s_idx] |= (1 << b)
        return current_G

    # Perform a swap
    def do_swap(e_rem, e_add, current_G, subset_masks, missing_edges, present_edges):
        # Remove e_rem
        current_G &= ~(1 << e_rem)
        present_edges.remove(e_rem)
        missing_edges.append(e_rem)
        for s_idx in edge_to_subsets[e_rem]:
            b_rem = local_idx[s_idx][e_rem]
            subset_masks[s_idx] &= ~(1 << b_rem)
        
        # Add e_add
        current_G |= (1 << e_add)
        missing_edges.remove(e_add)
        present_edges.append(e_add)
        for s_idx in edge_to_subsets[e_add]:
            b_add = local_idx[s_idx][e_add]
            subset_masks[s_idx] |= (1 << b_add)
        return current_G

    best_G = 0
    best_count = 0
    best_subset_masks = [0] * num_subsets
    
    # Phase 1: Multiple Random Greedy Restarts with varied strategies
    while time.time() < deadline - 40:
        edges = list(range(total_edges))
        strategy = random.random()
        if strategy < 0.3:
            random.shuffle(edges)
        elif strategy < 0.6:
            # Sort by some heuristic - e.g., edges that appear in fewer subsets first
            edges.sort(key=lambda e: len(edge_to_subsets[e]), reverse=True)
            random.shuffle(edges)
        else:
            # Random but with some structure
            for i in range(len(edges) - 1):
                j = random.randint(i, len(edges) - 1)
                edges[i], edges[j] = edges[j], edges[i]
        
        G, subset_masks = greedy_construct(edges)
        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G
            best_subset_masks = subset_masks[:]

    # Phase 2: Intensive Local Search
    current_G = best_G
    current_subset_masks = best_subset_masks[:]
    current_count = best_count
    
    # Maintain list of missing edges for fast random selection
    missing_edges = [i for i in range(total_edges) if not (current_G >> i) & 1]
    present_edges = [i for i in range(total_edges) if (current_G >> i) & 1]
    
    # Intensive hill climbing with restarts
    while time.time() < deadline - 10:
        improved = False
        
        # Try multiple add attempts
        for _ in range(20):
            if not missing_edges:
                break
            e_add = find_valid_add(missing_edges, current_subset_masks, max_tries=10)
            if e_add != -1:
                current_G = do_add(e_add, current_G, current_subset_masks, missing_edges, present_edges)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_G = current_G
                    best_subset_masks = current_subset_masks[:]
                improved = True
                break
        
        # If no simple add, try swaps
        if not improved:
            e_rem, e_add = find_valid_swap(missing_edges, present_edges, current_subset_masks, max_tries=50)
            if e_add != -1:
                current_G = do_swap(e_rem, e_add, current_G, current_subset_masks, missing_edges, present_edges)
                # Count stays same, but structure may improve for future adds
        
        # If stuck, do a small perturbation and restart local search
        if not improved and present_edges:
            # Remove 1-2 random edges to escape local optimum
            num_rem = random.randint(1, 2)
            for _ in range(num_rem):
                if not present_edges:
                    break
                e_rem = random.choice(present_edges)
                current_G &= ~(1 << e_rem)
                present_edges.remove(e_rem)
                missing_edges.append(e_rem)
                for s_idx in edge_to_subsets[e_rem]:
                    b_rem = local_idx[s_idx][e_rem]
                    current_subset_masks[s_idx] &= ~(1 << b_rem)
                current_count -= 1
            
            # Try to rebuild
            for _ in range(10):
                if not missing_edges:
                    break
                e_add = find_valid_add(missing_edges, current_subset_masks, max_tries=5)
                if e_add != -1:
                    current_G = do_add(e_add, current_G, current_subset_masks, missing_edges, present_edges)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_G = current_G
                        best_subset_masks = current_subset_masks[:]

    # Phase 3: Final polishing - try all possible single adds
    if time.time() < deadline - 2:
        current_G = best_G
        current_subset_masks = best_subset_masks[:]
        current_count = best_count
        missing_edges = [i for i in range(total_edges) if not (current_G >> i) & 1]
        present_edges = [i for i in range(total_edges) if (current_G >> i) & 1]
        
        # Systematically try each missing edge
        for e_add in list(missing_edges):
            if time.time() >= deadline - 2:
                break
            if can_add_edge(current_G, e_add, current_subset_masks):
                current_G = do_add(e_add, current_G, current_subset_masks, missing_edges, present_edges)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_G = current_G
                    best_subset_masks = current_subset_masks[:]

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result

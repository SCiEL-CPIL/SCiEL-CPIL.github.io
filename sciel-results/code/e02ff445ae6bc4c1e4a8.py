import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118

    n = 16

    # Edge indexing
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

    # All 8-subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)

    # Local K8 edge indices
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1

    # Q3 masks in K8: a Q3 is a set of 12 edges forming a cube
    # Cube: vertices 0..7, edges connect vertices differing in exactly one bit
    # We enumerate all 3-cube subgraphs in K8
    q3_masks_for_k8 = []
    seen_masks = set()
    # A Q3 in K8: pick 8 vertices, they form a cube if their induced subgraph has exactly 12 edges
    # and each vertex has degree 3 within the subset.
    # Equivalently: the 8 vertices form a "linear" structure.
    # A cube Q3 has 8 vertices, 12 edges. In K8, a subset of 8 vertices is a cube iff
    # the induced subgraph is 3-regular with 12 edges and is bipartite with parts of size 4.
    # Actually, the simplest: enumerate all 8-subsets (which is all of them since we have 8 vertices)
    # and check which 12-edge subsets form a cube.
    # Better: a cube is determined by its 8 vertices and the 3 "directions".
    # In K8 with vertices 0..7, a Q3 subgraph is a set of 12 edges such that the graph is 3-regular and bipartite.
    # Let's just enumerate all 12-edge subsets of K8 and check if they form a cube.
    # That's C(28,12) which is too many.
    # Instead: a Q3 is a 3-dimensional hypercube. Its vertices can be labeled 0..7 (3-bit).
    # Two vertices are adjacent iff they differ in exactly one bit.
    # So in K8, pick a bijection from {0..7} to the 8 vertices, and the edges are those pairs
    # whose labels differ in exactly one bit.
    # Number of distinct Q3 subgraphs in K8: 8! / (symmetries) ... but we just need all edge sets.
    # Two labelings give the same edge set iff they differ by a cube automorphism (16 of them).
    # So there are 8!/16 = 2520 distinct Q3 edge sets in K8.
    # Let's generate them.
    for perm in itertools.permutations(range(8)):
        mask = 0
        # perm[i] = vertex in K8 corresponding to cube vertex i
        for i in range(8):
            for j in range(i+1, 8):
                # Check if i and j differ in exactly one bit
                d = i ^ j
                if d & (d - 1) == 0:  # power of 2
                    u = perm[i]
                    v = perm[j]
                    e_idx = local_edge_idx[u][v]
                    mask |= (1 << e_idx)
        if mask not in seen_masks:
            seen_masks.add(mask)
            q3_masks_for_k8.append(mask)

    # Map global edges to subsets
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1]*total_edges for _ in range(num_subsets)]
    local_to_global = [[-1]*28 for _ in range(num_subsets)]

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

    # Q3 indices by local edge position
    q3_indices_by_local_edge = [[] for _ in range(28)]
    for q_idx, mask in enumerate(q3_masks_for_k8):
        for l in range(28):
            if (mask >> l) & 1:
                q3_indices_by_local_edge[l].append(q_idx)

    # For each subset, precompute: for each local edge, which Q3s contain it
    # (already done via q3_indices_by_local_edge, which is global to local positions)

    # Precompute for each subset and each pair of local edges, the Q3s that contain both
    # This helps for swap checks
    # Actually, let's use a different approach for speed:
    # For each subset, maintain the set of Q3s that are "complete" (all edges present)
    # When we add edge e to subset s: check if any Q3 containing e becomes complete
    # When we remove edge e from subset s: check if any Q3 containing e was complete (and now isn't)

    # For faster checking, for each subset s and local edge l, store the list of Q3 indices
    # (this is q3_indices_by_local_edge[l] which is the same for all subsets since local positions are consistent)

    # Actually the local positions are the same across subsets (0..27), so q3_indices_by_local_edge works directly.

    # For each subset, track the count of how many edges of each Q3 are present
    # This allows O(1) check for whether adding an edge completes a Q3
    # q3_edge_count[s_idx][q_idx] = number of edges of Q3 q_idx present in subset s_idx

    # This is 1144 * 2520 = ~2.9M integers, which is fine memory-wise but let's check
    # Actually, let's use a more compact representation.
    # For each subset, we have 28 edges. For each Q3 (12 edges), we need to count how many are present.
    # Let's precompute for each subset and each Q3, the bitmask of local edges in the Q3.
    # Then the count is popcount(subset_mask & q3_mask).
    # We already have q3_masks_for_k8 as bitmasks over local edges.
    # So: count = (subset_masks[s_idx] & q3_masks_for_k8[q_idx]).bit_count()
    # Adding edge l: new_count = old_count + 1 if l in Q3, else old_count
    # We need to check if new_count == 12 for any Q3 containing l.

    # For speed, let's precompute for each local edge l, the list of Q3s containing it (done)
    # and for each such Q3, the mask (done).

    def can_add_edge(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            for q_idx in q3_indices_by_local_edge[l_pos]:
                if (new_mask & q3_masks_for_k8[q_idx]) == q3_masks_for_k8[q_idx]:
                    return False
        return True

    def find_blocking_q3(G, e_idx, subset_masks):
        blocking_edges = []
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            for q_idx in q3_indices_by_local_edge[l_pos]:
                q_mask = q3_masks_for_k8[q_idx]
                if (new_mask & q_mask) == q_mask:
                    for l in range(28):
                        if (q_mask >> l) & 1:
                            g_idx = local_to_global[s_idx][l]
                            if g_idx != -1 and (G >> g_idx) & 1:
                                blocking_edges.append(g_idx)
                    break
        return blocking_edges

    def try_swap(G, e_add, e_remove, subset_masks):
        """Check if removing e_remove and adding e_add keeps the graph Q3-free."""
        # Remove e_remove from subset_masks
        for s_idx in edge_to_subsets[e_remove]:
            l_pos_r = local_idx[s_idx][e_remove]
            subset_masks[s_idx] &= ~(1 << l_pos_r)
        
        # Try to add e_add
        valid = True
        for s_idx in edge_to_subsets[e_add]:
            l_pos_a = local_idx[s_idx][e_add]
            new_mask = subset_masks[s_idx] | (1 << l_pos_a)
            for q_idx in q3_indices_by_local_edge[l_pos_a]:
                if (new_mask & q3_masks_for_k8[q_idx]) == q3_masks_for_k8[q_idx]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Commit the add
            for s_idx in edge_to_subsets[e_add]:
                l_pos_a = local_idx[s_idx][e_add]
                subset_masks[s_idx] |= (1 << l_pos_a)
            return True
        else:
            # Rollback: re-add e_remove
            for s_idx in edge_to_subsets[e_remove]:
                l_pos_r = local_idx[s_idx][e_remove]
                subset_masks[s_idx] |= (1 << l_pos_r)
            return False

    def try_double_swap(G, e_add1, e_add2, e_remove1, e_remove2, subset_masks):
        """Check if removing e_remove1, e_remove2 and adding e_add1, e_add2 keeps Q3-free."""
        for s_idx in edge_to_subsets[e_remove1]:
            l_pos_r1 = local_idx[s_idx][e_remove1]
            subset_masks[s_idx] &= ~(1 << l_pos_r1)
        for s_idx in edge_to_subsets[e_remove2]:
            l_pos_r2 = local_idx[s_idx][e_remove2]
            subset_masks[s_idx] &= ~(1 << l_pos_r2)
        
        valid = True
        for e_add in (e_add1, e_add2):
            for s_idx in edge_to_subsets[e_add]:
                l_pos_a = local_idx[s_idx][e_add]
                new_mask = subset_masks[s_idx] | (1 << l_pos_a)
                for q_idx in q3_indices_by_local_edge[l_pos_a]:
                    if (new_mask & q3_masks_for_k8[q_idx]) == q3_masks_for_k8[q_idx]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            for e_add in (e_add1, e_add2):
                for s_idx in edge_to_subsets[e_add]:
                    l_pos_a = local_idx[s_idx][e_add]
                    subset_masks[s_idx] |= (1 << l_pos_a)
            return True
        else:
            for s_idx in edge_to_subsets[e_remove1]:
                l_pos_r1 = local_idx[s_idx][e_remove1]
                subset_masks[s_idx] |= (1 << l_pos_r1)
            for s_idx in edge_to_subsets[e_remove2]:
                l_pos_r2 = local_idx[s_idx][e_remove2]
                subset_masks[s_idx] |= (1 << l_pos_r2)
            return False

    best_G = 0
    best_count = 0

    # Initial good starting point: try a few different greedy strategies
    # Strategy: add edges in order of how "central" they are (high degree vertices first)
    # Or just use random restarts with different seeds

    for restart in range(50):
        if time.time() > deadline:
            break
        
        G = 0
        subset_masks = [0] * num_subsets
        missing_edges = set(range(total_edges))
        
        # Greedy Phase with randomized order
        edges = list(range(total_edges))
        random.shuffle(edges)
        
        for e_idx in edges:
            if time.time() > deadline:
                break
            if can_add_edge(G, e_idx, subset_masks):
                G |= (1 << e_idx)
                missing_edges.remove(e_idx)
                for s_idx in edge_to_subsets[e_idx]:
                    l_pos = local_idx[s_idx][e_idx]
                    subset_masks[s_idx] |= (1 << l_pos)
        
        # Local Search Phase
        improved = True
        stall_count = 0
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing edge
            missing_list = list(missing_edges)
            random.shuffle(missing_list)
            
            for e_idx in missing_list:
                if can_add_edge(G, e_idx, subset_masks):
                    G |= (1 << e_idx)
                    missing_edges.remove(e_idx)
                    for s_idx in edge_to_subsets[e_idx]:
                        l_pos = local_idx[s_idx][e_idx]
                        subset_masks[s_idx] |= (1 << l_pos)
                    improved = True
                    break
            
            # 2. If no add, try a single swap
            if not improved and missing_edges:
                e_idx = random.choice(list(missing_edges))
                blocking_edges = find_blocking_q3(G, e_idx, subset_masks)
                if blocking_edges:
                    random.shuffle(blocking_edges)
                    for f_idx in blocking_edges:
                        if try_swap(G, e_idx, f_idx, subset_masks):
                            G = (G & ~(1 << f_idx)) | (1 << e_idx)
                            missing_edges.remove(e_idx)
                            missing_edges.add(f_idx)
                            improved = True
                            break
            
            # 3. If no single swap, try a double swap
            if not improved and len(missing_edges) >= 2:
                missing_list = list(missing_edges)
                random.shuffle(missing_list)
                e1 = missing_list[0]
                e2 = missing_list[1]
                b1 = find_blocking_q3(G, e1, subset_masks)
                b2 = find_blocking_q3(G, e2, subset_masks)
                if b1 and b2:
                    random.shuffle(b1)
                    random.shuffle(b2)
                    for f1 in b1[:5]:
                        for f2 in b2[:5]:
                            if f1 == f2:
                                continue
                            if try_double_swap(G, e1, e2, f1, f2, subset_masks):
                                G = (G & ~(1 << f1) & ~(1 << f2)) | (1 << e1) | (1 << e2)
                                missing_edges.remove(e1)
                                missing_edges.remove(e2)
                                missing_edges.add(f1)
                                missing_edges.add(f2)
                                improved = True
                                break
                        if improved:
                            break

        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G

    # Final verification and output
    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result

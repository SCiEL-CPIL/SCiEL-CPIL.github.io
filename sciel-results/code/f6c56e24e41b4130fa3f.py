import time
import random
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    # Precompute edge indices for all pairs
    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)

    # Inverse map: index -> (u, v)
    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Local edge index mapping for K8
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    # Q3 abstract edges
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Precompute all local Q3 edge sets (as frozensets of local edge indices)
    # There are only 8! = 40320 permutations, which is fast.
    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))

    # Convert to list of bitmasks
    local_q3_bitmasks = []
    for mask_set in local_q3_edge_sets:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)

    # Precompute global Q3 masks
    # For each subset of 8 vertices, map local edges to global edges
    # and generate all Q3 masks for that subset.
    
    # Optimization: Precompute for each subset the mapping from local edge idx to global edge idx
    # We will store a list of global Q3 masks.
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    # Precompute local edge (u,v) -> local_idx for K8
    local_edge_to_idx = {}
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = local_edge_index(u, v)

    # Precompute the set of local edges in each local Q3 bitmask?
    # No, we already have the bitmasks.
    # We need to iterate over subsets.
    # 12870 subsets. For each, we need to build the global mask for each local Q3.
    # This is 12870 * 840 = ~10.8M operations.
    # Each operation involves 12 edge lookups and bit shifts.
    # This might be slow in pure Python. Let's see if we can optimize.
    
    # Alternative: Iterate over all permutations of 8 vertices from 0..15?
    # That's P(16, 8) which is huge.
    # So we must use the subset approach.
    
    # Let's precompute the local edge indices for K8 as a list of tuples
    # and map them to global indices quickly.
    
    # We can precompute for each subset, a list of 28 global edge indices corresponding to local edges 0..27.
    # Then for each local Q3 bitmask, we OR the corresponding global bits.
    
    # To speed up the inner loop, we can precompute for each local Q3 bitmask, the list of local edge indices it contains.
    local_q3_edge_lists = []
    for mask in local_q3_bitmasks:
        edges = []
        for i in range(28):
            if mask & (1 << i):
                edges.append(i)
        local_q3_edge_lists.append(edges)

    # Now, for each subset:
    for subset in all_subsets:
        # Map local edge idx to global edge idx
        # local edge idx i corresponds to a pair (u_loc, v_loc).
        # We need a mapping from local edge idx to the global edge idx.
        # Let's create an array l2g of size 28.
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        # For each local Q3, generate the global mask
        for edge_list in local_q3_edge_lists:
            global_mask = 0
            for le in edge_list:
                ge = l2g[le]
                global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    num_q3s = len(global_q3_masks)
    if time.time() > deadline:
        # Fallback: return a simple valid graph (e.g., empty or sparse)
        # But we need to return *something* valid.
        # A random sparse graph is likely valid.
        # Let's just return edges of a cycle or something.
        # Actually, if we time out here, we haven't even started the greedy.
        # Let's just return a simple bipartite graph K8,8 which is Q3-free?
        # Q3 is bipartite, so K8,8 might contain it?
        # K3,3 is a subgraph of K8,8. Q3 has 8 vertices.
        # It's safer to return a very sparse graph.
        # Let's return a matching.
        return [[i, i+1] for i in range(0, 16, 2)]

    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    
    # Build lists: q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # Initialize counts
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    removed_edges = [False] * num_edges_total
    
    # Greedy Loop
    while True:
        if time.time() > deadline:
            break
            
        # Find edge with max count
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_edges[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        
        # Update active Q3s and counts
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # Local search polish: try to re-add removed edges if the graph stays Q3-free
    # Build the current edge set as a bitmask for quick lookup
    current_mask = 0
    for e in range(num_edges_total):
        if not removed_edges[e]:
            current_mask |= (1 << e)
    
    # For each removed edge, check if adding it back creates a Q3
    # We need to check: does there exist a Q3 that uses this edge and all other edges in current_mask?
    # A Q3 containing edge e is "hit" if at least one of its 12 edges is missing from current_mask.
    # Adding edge e back is safe if no Q3 containing e has all its other 11 edges present.
    
    # Precompute for each Q3 the list of edges it contains (already have global_q3_masks)
    # For efficiency, for each Q3, store the set of edge indices
    
    # Build a list of Q3s per edge (already have q3s_containing_edge)
    # For each removed edge, try to re-add it
    
    # Sort removed edges by how many Q3s they hit (fewer first = safer to re-add)
    removed_list = []
    for e in range(num_edges_total):
        if removed_edges[e]:
            removed_list.append(e)
    
    # Sort by number of Q3s containing this edge (ascending) - safer edges first
    removed_list.sort(key=lambda e: len(q3s_containing_edge[e]))
    
    # Try to re-add edges
    for e in removed_list:
        if time.time() > deadline:
            break
        
        # Check if re-adding e creates any Q3
        # A Q3 is created if there exists a Q3 containing e where all other 11 edges are in current_mask
        safe = True
        for q3_idx in q3s_containing_edge[e]:
            mask = global_q3_masks[q3_idx]
            # Check if all edges in this Q3 except e are in current_mask
            # i.e., (mask & ~(1 << e)) should be a subset of current_mask
            other_edges = mask & ~(1 << e)
            if (other_edges & ~current_mask) == 0:
                # This Q3 would be completed if we add e
                safe = False
                break
        
        if safe:
            # Re-add this edge
            removed_edges[e] = False
            current_mask |= (1 << e)
    
    # The remaining edges are the solution
    result_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))

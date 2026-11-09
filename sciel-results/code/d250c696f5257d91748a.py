import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

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

    # Q3 abstract edges (standard labeling of 3-cube)
    # Vertices 0..7. Edges:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Local edge index mapping for K8 (28 edges)
    local_edge_to_idx = {}
    local_idx_to_edge = [None] * 28
    idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = idx
            local_idx_to_edge[idx] = (u, v)
            idx += 1

    # Precompute local Q3 edge sets as bitmasks of 28 bits
    # We iterate over all permutations of 8 vertices to cover all isomorphisms
    # However, many permutations yield the same edge set. We use a set of frozensets.
    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_to_idx[(min(u_val, v_val), max(u_val, v_val))]
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))
    
    # Convert to list of bitmasks for faster operations
    local_q3_bitmasks = []
    for mask_set in local_q3_edge_sets:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)

    # Precompute global Q3 masks
    # For each subset of 8 vertices, map local edges to global edges
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    if time.time() > deadline - 20:
        # If precomputation is too slow, fallback to a simple heuristic
        return [[i, i+1] for i in range(0, 16, 2)]

    # To speed up global mask generation, we can precompute for each local Q3 bitmask,
    # the list of local edge indices it contains.
    local_q3_edge_lists = []
    for mask in local_q3_bitmasks:
        edges = []
        m = mask
        while m:
            lsb = m & -m
            edges.append(lsb.bit_length() - 1)
            m ^= lsb
        local_q3_edge_lists.append(edges)

    for subset in all_subsets:
        # Map local edge idx to global edge idx
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
            
    if time.time() > deadline - 10:
        # Fallback if precomputation took too long
        return [[i, i+1] for i in range(0, 16, 2)]

    num_q3s = len(global_q3_masks)
    
    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    # removed_edges[e] = True if edge e is removed (not in graph)
    removed_edges = [False] * num_edges_total
    
    # q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # counts[e] = number of active Q3s containing e
    counts = [0] * num_edges_total
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    # active_q3s: boolean list, True if Q3 is not yet hit
    active_q3s = [True] * num_q3s
    
    # Greedy Loop
    while True:
        if time.time() > deadline - 20:
            break
            
        # Find edge with max count among not-removed edges
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
            if active_q3s[q3_idx]:
                active_q3s[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # Local Search Polish: try to re-add removed edges
    # We iterate through removed edges. If adding edge e back does not create a Q3, we add it.
    # We maintain a count for each Q3 of how many of its edges are removed.
    # If count == 0, the Q3 is present in the graph.
    
    # Initialize removed_count_for_q3
    removed_count_for_q3 = [0] * num_q3s
    for q3_idx, mask in enumerate(global_q3_masks):
        m = mask
        cnt = 0
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            if removed_edges[e_idx]:
                cnt += 1
            m ^= lsb
        removed_count_for_q3[q3_idx] = cnt
    
    # Also maintain for each edge, the list of Q3s it belongs to (already have q3s_containing_edge)
    
    # Try to add back removed edges
    # Order: random or by some heuristic? Let's try random first, then maybe deterministic.
    # We can do multiple passes.
    
    # For efficiency, we can just iterate through all removed edges in a random order and try to add them.
    # If adding e makes any Q3 have removed_count == 0, we reject.
    # When we add e (set removed_edges[e] = False), we decrement removed_count_for_q3 for all q3 in q3s_containing_edge[e].
    # We check if any becomes 0. If so, we revert (increment back) and skip.
    
    # To make it stronger, we can do multiple passes, shuffling the order of removed edges.
    # Or we can sort removed edges by some priority (e.g., degree in the removed graph, or just random).
    
    # Let's do a few passes with random shuffles.
    removed_list = [e for e in range(num_edges_total) if removed_edges[e]]
    
    for pass_num in range(10):
        if time.time() > deadline - 5:
            break
        random.seed(pass_num * 42 + 123)
        random.shuffle(removed_list)
        
        changed = False
        for e in removed_list:
            # Check if e is still removed
            if not removed_edges[e]:
                continue
            
            # Try to add e
            # Decrement counts for all Q3s containing e
            affected_q3s = q3s_containing_edge[e]
            made_zero = False
            for q3_idx in affected_q3s:
                removed_count_for_q3[q3_idx] -= 1
                if removed_count_for_q3[q3_idx] == 0:
                    made_zero = True
                    break
            
            if made_zero:
                # Revert
                for q3_idx in affected_q3s:
                    removed_count_for_q3[q3_idx] += 1
            else:
                # Accept
                removed_edges[e] = False
                changed = True
        
        if not changed:
            break
    
    # Construct the final edge list
    final_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            final_edges.append([u, v])
            
    return final_edges

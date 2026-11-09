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

    # Precompute local Q3 masks on 8 vertices (0..7)
    # Q3 edges in abstract space (using standard labeling 000..111)
    q3_abstract_edges = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (3,7),
        (4,5), (4,6),
        (5,7),
        (6,7)
    ]
    
    # Local edge index mapping for K8
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    local_q3_masks = set()
    
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        
        if len(edges_set) == 12:
            local_q3_masks.add(frozenset(edges_set))
            
    local_q3_bitmasks = []
    for mask_set in local_q3_masks:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)
        
    # Generate global Q3 masks
    global_q3_masks = []
    
    all_subsets = list(combinations(range(n), 8))
    
    for subset in all_subsets:
        l2g_edge = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g_edge[le] = ge
                
        for local_mask in local_q3_bitmasks:
            global_mask = 0
            for le in range(28):
                if local_mask & (1 << le):
                    ge = l2g_edge[le]
                    global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    num_q3s = len(global_q3_masks)
    
    # Precompute for each edge, which Q3s it belongs to
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
    
    # --- Phase 1: Greedy Deletion ---
    # We start with K16 (all edges present) and delete edges to eliminate all Q3s.
    # Heuristic: Always delete the edge that is part of the most remaining Q3s.
    
    # counts[e] = number of active Q3s that contain edge e
    counts = [len(q3s_containing_edge[e]) for e in range(num_edges_total)]
    active_q3 = [True] * num_q3s
    present = [True] * num_edges_total
    
    removed_edges = []
    
    # To speed up finding max count, we can just scan. 120 is small.
    # But let's see if we can be smarter.
    
    while True:
        if time.time() > deadline - 5.0: # Leave 5s for polish
            break
            
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if not present[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Delete best_edge
        present[best_edge] = False
        removed_edges.append(best_edge)
        
        # Update counts for other edges in the Q3s that are now destroyed
        for q3_idx in q3s_containing_edge[best_edge]:
            if active_q3[q3_idx]:
                active_q3[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge and present[e_idx]:
                        counts[e_idx] -= 1
                    m ^= lsb

    # Now we have a valid graph (no Q3s).
    # Current mask:
    curr_mask = 0
    for e in range(num_edges_total):
        if present[e]:
            curr_mask |= (1 << e)
            
    curr_edges = sum(present)
    
    # --- Phase 2: Local Search (Add Edges) ---
    # Try to add back edges.
    # An edge e can be added if it doesn't complete any Q3.
    # i.e., for all q3 in q3s_containing_edge[e], at least one other edge in q3 is missing.
    
    # We can check this efficiently if we track the number of "missing" edges in each Q3.
    # Let's define:
    # missing_count[q3] = number of edges in q3 that are NOT present.
    # Initially, missing_count[q3] = 12 - (number of present edges in q3).
    # If we add an edge e, for each q3 containing e, missing_count[q3] decreases by 1.
    # If missing_count[q3] becomes 0, we have a violation.
    # So we can only add e if for all q3 in q3s_containing_edge[e], missing_count[q3] > 1.
    # (Because adding e reduces it by 1, so it must be >= 2 before adding to stay >= 1 after).
    
    # Let's compute initial missing counts.
    missing_counts = [0] * num_q3s
    for i, mask in enumerate(global_q3_masks):
        m = mask
        missing = 0
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            if not present[e_idx]:
                missing += 1
            m ^= lsb
        missing_counts[i] = missing
        
    # Now try to add edges.
    # We can use a greedy approach: repeatedly find an edge that can be added.
    # To maximize density, we might want to add edges that "unlock" other edges?
    # Or just random greedy.
    
    # Let's try a randomized greedy approach with restarts or just a single pass?
    # Given the time limit, let's do a simple loop:
    # While time remains:
    #   Find a random edge that can be added.
    #   Add it.
    #   If none, try to remove a random edge and see if it allows more additions? (Too complex)
    #   Or just stop.
    
    # Actually, the order of addition matters.
    # Let's try to add edges in a random order, skipping those that would cause a violation.
    
    # Shuffle edges that are currently missing.
    missing_edge_list = [e for e in range(num_edges_total) if not present[e]]
    random.shuffle(missing_edge_list)
    
    # Try to add them
    for e in missing_edge_list:
        if time.time() > deadline - 2.0:
            break
            
        # Check if e can be added
        can_add = True
        for q3_idx in q3s_containing_edge[e]:
            if missing_counts[q3_idx] <= 1:
                can_add = False
                break
                
        if can_add:
            present[e] = True
            curr_edges += 1
            # Update missing counts
            for q3_idx in q3s_containing_edge[e]:
                missing_counts[q3_idx] -= 1

    # Now we have a valid graph with curr_edges edges.
    # Can we do better?
    # The greedy addition order is random. Maybe a different order yields more edges.
    # Let's try a few restarts with different random seeds/orders if time permits.
    
    best_edges_found = curr_edges
    best_present = present[:]
    best_missing_counts = missing_counts[:]
    
    # We have ~100s left. Let's do multiple restarts of the local search phase.
    # But restarting the whole greedy deletion is expensive.
    # Can we restart just the addition phase?
    # Yes, we have the state after deletion.
    
    # Let's save the state after deletion.
    base_present = present[:]
    base_missing_counts = missing_counts[:]
    base_edges = curr_edges
    
    # We already did one addition pass.
    # Let's try more passes with different random shuffles.
    
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        # Restore base state
        present = base_present[:]
        missing_counts = base_missing_counts[:]
        curr_edges = base_edges
        
        missing_edge_list = [e for e in range(num_edges_total) if not present[e]]
        random.shuffle(missing_edge_list)
        
        for e in missing_edge_list:
            if time.time() > deadline - 5.0:
                break
            
            can_add = True
            for q3_idx in q3s_containing_edge[e]:
                if missing_counts[q3_idx] <= 1:
                    can_add = False
                    break
                    
            if can_add:
                present[e] = True
                curr_edges += 1
                for q3_idx in q3s_containing_edge[e]:
                    missing_counts[q3_idx] -= 1
                    
        if curr_edges > best_edges_found:
            best_edges_found = curr_edges
            best_present = present[:]
            best_missing_counts = missing_counts[:]

    # Final Polish: Try to add any remaining edges that can be added (deterministic sweep)
    # Iterate through all missing edges, try to add them. Repeat until no more can be added.
    # This is a "saturation" step.
    
    # Use the best state found
    present = best_present[:]
    missing_counts = best_missing_counts[:]
    
    improved = True
    while improved:
        improved = False
        for e in range(num_edges_total):
            if present[e]:
                continue
            can_add = True
            for q3_idx in q3s_containing_edge[e]:
                if missing_counts[q3_idx] <= 1:
                    can_add = False
                    break
            if can_add:
                present[e] = True
                improved = True
                for q3_idx in q3s_containing_edge[e]:
                    missing_counts[q3_idx] -= 1

    # Construct result
    result_edges = []
    for e in range(num_edges_total):
        if present[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))

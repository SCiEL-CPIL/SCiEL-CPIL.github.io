import time
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    d = 7
    n = 2 ** d
    total_edges = n * d // 2  # 448
    
    # 1. Generate all edges and map to indices
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx
    
    # 2. Precompute C4 structures
    # A C4 is defined by 2 bits i, j and a base vertex x where bit i, j are 0.
    c4_edges_list = []
    c4_membership = [[] for _ in range(total_edges)]
    
    for i in range(d):
        for j in range(i+1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for x in range(n):
                if (x & mask_i) == 0 and (x & mask_j) == 0:
                    v1 = x
                    v2 = x ^ mask_i
                    v3 = x ^ mask_j
                    v4 = x ^ mask_i ^ mask_j
                    
                    e1 = edge_index.get((min(v1, v2), max(v1, v2)))
                    e2 = edge_index.get((min(v3, v4), max(v3, v4)))
                    e3 = edge_index.get((min(v1, v3), max(v1, v3)))
                    e4 = edge_index.get((min(v2, v4), max(v2, v4)))
                    
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        c4_idx = len(c4_edges_list)
                        c4_edges_list.append((e1, e2, e3, e4))
                        c4_membership[e1].append(c4_idx)
                        c4_membership[e2].append(c4_idx)
                        c4_membership[e3].append(c4_idx)
                        c4_membership[e4].append(c4_idx)
                        
    num_c4s = len(c4_edges_list)
    # Convert to numpy arrays for vectorized operations
    c4_edges = np.array(c4_edges_list, dtype=np.int32) # shape (num_c4s, 4)
    # Incidence matrix: incidence[e, c] = 1 if edge e is in C4 c
    incidence = np.zeros((total_edges, num_c4s), dtype=np.int32)
    for c_idx in range(num_c4s):
        for e_idx in c4_edges[c_idx]:
            incidence[e_idx, c_idx] = 1

    best_m = 0
    best_present = None

    # Helper to initialize "bad_count" for an edge
    # bad_count[e] = number of C4s containing e where the other 3 edges are present
    def compute_bad_counts(present):
        bad_counts = np.zeros(total_edges, dtype=np.int32)
        for c_idx in range(num_c4s):
            edges_in_c4 = c4_edges[c_idx]
            if present[edges_in_c4].all():
                # All 4 are present, so for each edge, the other 3 are present
                for e_idx in edges_in_c4:
                    bad_counts[e_idx] += 1
            else:
                # Check each edge in this C4
                # If edge e is present, check if the other 3 are present
                # If edge e is NOT present, it doesn't contribute to bad_counts for itself
                # but its presence/absence affects the count for others?
                # No, bad_counts[e] depends on the state of the OTHER edges in C4s containing e.
                # Let's just compute it directly for simplicity in initialization,
                # but for updates we will do incremental.
                pass
        
        # More efficient initialization:
        # For each C4, if it's not full, we can check which edges are "blocked" by it.
        # An edge e is blocked by C4 c if e is in c and the other 3 edges of c are present.
        # Let's re-calculate from scratch for correctness in init, it's fast enough (448 * 21 ~ 9k ops)
        bad_counts = np.zeros(total_edges, dtype=np.int32)
        for c_idx in range(num_c4s):
            edges_in_c4 = c4_edges[c_idx]
            # If all 4 are present, all 4 edges have 3 neighbors present in this C4
            if present[edges_in_c4].all():
                for e_idx in edges_in_c4:
                    bad_counts[e_idx] += 1
            else:
                # Count how many are present
                # If exactly 3 are present, the 1 missing edge has 3 neighbors present? No.
                # If 3 are present, the missing edge e has 3 neighbors present in this C4.
                # So bad_counts[e] += 1 for the missing edge.
                # If 2 are present, no edge has 3 neighbors present.
                # If 1 or 0, none.
                present_count = np.sum(present[edges_in_c4])
                if present_count == 3:
                    # Find the missing edge
                    for e_idx in edges_in_c4:
                        if not present[e_idx]:
                            bad_counts[e_idx] += 1
                            break
        return bad_counts

    # Local Search with incremental updates
    def local_search(present, bad_counts, m, time_limit):
        improved = True
        removed_indices = np.where(~present)[0]
        
        # Shuffle removed indices for randomization
        np.random.shuffle(removed_indices)
        
        while improved and time.time() < time_limit:
            improved = False
            for e in removed_indices:
                if present[e]:
                    continue
                # If bad_counts[e] == 0, we can add e
                if bad_counts[e] == 0:
                    # Add edge e
                    present[e] = True
                    m += 1
                    improved = True
                    
                    # Update bad_counts for all edges in C4s containing e
                    # For each C4 c containing e:
                    # 1. The other 3 edges in c now have one more neighbor present (e).
                    #    So their bad_count increases by 1 IF the other 2 neighbors were already present.
                    #    Actually, bad_count[x] counts C4s where the other 3 are present.
                    #    Before adding e: For x in c\{e}, if the other 2 in c\{e,x} were present,
                    #    x was NOT blocked by c (because e was missing). Now e is present.
                    #    So if the other 2 are present, x becomes blocked by c.
                    #    So bad_count[x] += 1 if sum(present[c\{e,x}]) == 2.
                    # 2. For e itself: bad_count[e] was 0. Now e is present.
                    #    Does e become blocked? No, e is present. We only track bad_counts for edges we might add?
                    #    Actually, we only use bad_counts for edges NOT in present.
                    #    But when we remove edges later, we need accurate counts.
                    #    Let's maintain bad_counts for ALL edges.
                    #    If e is present, bad_counts[e] is irrelevant for adding e, but might be relevant if we remove e later?
                    #    No, if we remove e, we revert.
                    #    However, the definition "other 3 are present" applies regardless of e's state?
                    #    Usually, we define "conflict" for an edge e as: e is not present, and there exists a C4 containing e where the other 3 are present.
                    #    If e IS present, it's part of a C4. If the other 3 are present, the C4 is "bad" (all 4 present).
                    #    Our algorithm ensures we never have a C4 with all 4 present.
                    #    So if e is present, for any C4 containing e, the other 3 cannot all be present.
                    #    Therefore, bad_counts for present edges should be 0?
                    #    Let's verify: If we add e, we ensure bad_counts[e] was 0.
                    #    This means for all c containing e, the other 3 were NOT all present.
                    #    So after adding e, no C4 containing e has all 4 present.
                    #    So the "badness" of neighbors changes.
                    
                    for c_idx in c4_membership[e]:
                        edges_in_c4 = c4_edges[c_idx]
                        # Find the other 3 edges
                        others = [x for x in edges_in_c4 if x != e]
                        # Check how many of the other 3 are present
                        # If 2 are present, then adding e makes the 3rd one blocked?
                        # No. The 3rd one (say x) was not blocked before (because e was missing).
                        # Now e is present. The other two (y, z) are present.
                        # So for x, the other 3 in this C4 are e, y, z. All are present.
                        # So x becomes blocked. bad_counts[x] += 1.
                        # If 3 are present (impossible, would mean C4 was bad before adding e? No, e was missing.
                        # If y,z,w are present, and e missing, C4 is not bad.
                        # When we add e, C4 becomes bad.
                        # This means y, z, w are all blocked by this C4?
                        # Yes. bad_counts[y] += 1, bad_counts[z] += 1, bad_counts[w] += 1.
                        # But wait, if y,z,w are present, they are not "candidates" for addition.
                        # But we maintain bad_counts for all edges.
                        
                        num_present_others = np.sum(present[others])
                        
                        if num_present_others == 3:
                            # All 3 others are present. Adding e creates a bad C4.
                            # This shouldn't happen if bad_counts[e] was 0.
                            # bad_counts[e] == 0 means for all c containing e, NOT (all others present).
                            # So num_present_others must be < 3.
                            pass
                        elif num_present_others == 2:
                            # Two others are present.
                            # The third other (let's call it x) now has e, y, z present?
                            # No, the C4 has e, x, y, z.
                            # If y, z are present, and e is now present, then for x, the other 3 are e, y, z.
                            # So x becomes blocked.
                            # Find x
                            for x in others:
                                if not present[x]:
                                    bad_counts[x] += 1
                                    break
                        # If num_present_others == 0 or 1, no new blocks are created.
                        
                    # Also, does adding e affect bad_counts[e]?
                    # e is now present. We don't care about bad_counts[e] for addition.
                    # But if we remove e later, we need to revert.
                    # We'll handle revert symmetrically.
                    
                    # Remove e from removed_indices?
                    # We are iterating over a static list. We check `if present[e]: continue`.
                    # So it's fine.
        
        return present, bad_counts, m

    # Main Loop
    max_restarts = 5
    for r in range(max_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initialize with all edges present
        present = np.ones(total_edges, dtype=bool)
        
        # Vectorized Greedy Deletion
        while True:
            if time.time() > deadline - 10:
                break
            
            c4_bad = present[c4_edges].all(axis=1)
            
            if not np.any(c4_bad):
                break
            
            edge_bad_counts = incidence @ c4_bad
            max_count = np.max(edge_bad_counts)
            if max_count == 0:
                break
                
            candidates = np.where(edge_bad_counts == max_count)[0]
            best_e = candidates[random.randint(0, len(candidates) - 1)]
            
            present[best_e] = False
        
        if time.time() > deadline - 10:
            break
            
        m = np.sum(present)
        
        # Compute initial bad counts
        bad_counts = compute_bad_counts(present)
        
        # Local Search
        time_limit_for_ls = deadline - 5
        present, bad_counts, m = local_search(present, bad_counts, m, time_limit_for_ls)
        
        if m > best_m:
            best_m = m
            best_present = present.copy()
            
    # Fallback
    if best_present is None:
        present = np.ones(total_edges, dtype=bool)
        while True:
            c4_bad = present[c4_edges].all(axis=1)
            if not np.any(c4_bad):
                break
            edge_bad_counts = incidence @ c4_bad
            max_count = np.max(edge_bad_counts)
            if max_count == 0:
                break
            candidates = np.where(edge_bad_counts == max_count)[0]
            best_e = candidates[0]
            present[best_e] = False
        best_present = present
        best_m = np.sum(best_present)

    out = []
    for idx, (u, v) in enumerate(all_edges):
        if best_present[idx]:
            out.append([u, v])
            
    return out
# EVOLVE-BLOCK-END

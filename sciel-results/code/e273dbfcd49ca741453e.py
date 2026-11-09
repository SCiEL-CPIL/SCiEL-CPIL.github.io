"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    d = 7
    n = 1 << d  # 128
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

    # 2. Precompute edge info (u, v, bit)
    edge_info = []
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        bit = diff.bit_length() - 1
        edge_info.append((u, v, bit))

    # 3. Precompute all C4s (faces)
    all_c4s = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(n):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v3, v4), max(v3, v4))]
                    e3 = edge_index[(min(v1, v3), max(v1, v3))]
                    e4 = edge_index[(min(v2, v4), max(v2, v4))]
                    
                    all_c4s.append((e1, e2, e3, e4))

    num_c4s = len(all_c4s)
    
    # 4. Precompute for each edge, the list of C4 indices it belongs to
    c4_membership = [[] for _ in range(total_edges)]
    for c4_idx, (e1, e2, e3, e4) in enumerate(all_c4s):
        for e in (e1, e2, e3, e4):
            c4_membership[e].append(c4_idx)

    # 5. Greedy Deletion with dynamic scoring
    present = [True] * total_edges
    # For each C4, track how many of its 4 edges are present
    c4_present_count = [4] * num_c4s  # Initially all 4 edges present in each C4
    
    bad_c4s = set(range(num_c4s))
    
    while bad_c4s:
        if time.time() > deadline - 30:
            break
            
        # Score each present edge by number of bad C4s it's in
        max_count = -1
        best_e = -1
        for e in range(total_edges):
            if not present[e]:
                continue
            count = 0
            for c4_idx in c4_membership[e]:
                if c4_idx in bad_c4s:
                    count += 1
            if count > max_count:
                max_count = count
                best_e = e
                
        if best_e == -1 or max_count == 0:
            break
            
        # Delete best_e
        present[best_e] = False
        
        # Update C4 present counts
        for c4_idx in c4_membership[best_e]:
            c4_present_count[c4_idx] -= 1
            if c4_present_count[c4_idx] < 4:
                bad_c4s.discard(c4_idx)

    m = sum(present)
    
    # 6. Try to add back removed edges
    removed = [e for e in range(total_edges) if not present[e]]
    random.shuffle(removed)
    
    for e in removed:
        if time.time() > deadline - 10:
            break
        ok = True
        for c4_idx in c4_membership[e]:
            if c4_present_count[c4_idx] >= 3:
                ok = False
                break
        if ok:
            present[e] = True
            m += 1
            for c4_idx in c4_membership[e]:
                c4_present_count[c4_idx] += 1

    # 7. Multiple passes to add more edges
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        progress = False
        for e in removed:
            if present[e]:
                continue
            ok = True
            for c4_idx in c4_membership[e]:
                if c4_present_count[c4_idx] >= 3:
                    ok = False
                    break
            if ok:
                present[e] = True
                m += 1
                for c4_idx in c4_membership[e]:
                    c4_present_count[c4_idx] += 1
                progress = True
        if not progress:
            break

    # 8. Local search: try swapping - remove one present edge, add up to 2 removed edges
    # This can help escape local optima
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        
        # Try to find a beneficial swap: remove edge e_present, add edges e_rem1, e_rem2
        # such that the net change in bad C4s is negative or zero, and we gain an edge
        best_swap = None
        best_gain = 0
        
        present_list = [e for e in range(total_edges) if present[e]]
        removed_list = [e for e in range(total_edges) if not present[e]]
        
        if not present_list or not removed_list:
            break
        
        # Sample a subset for efficiency
        random.shuffle(present_list)
        random.shuffle(removed_list)
        
        sample_present = present_list[:min(50, len(present_list))]
        sample_removed = removed_list[:min(50, len(removed_list))]
        
        for e_pres in sample_present:
            # If we remove e_pres, which C4s become less bad?
            # For each C4 containing e_pres, if it had 4 present, now 3
            # But we also need to check if adding removed edges creates new bad C4s
            
            # Compute effect of removing e_pres: decreases c4_present_count for its C4s
            # Compute effect of adding e_rem: increases c4_present_count for its C4s
            
            # We want: after swap, no C4 has all 4 present
            # And we gain at least 1 edge (add 2, remove 1)
            
            for e_rem1 in sample_removed:
                if e_rem1 == e_pres:
                    continue
                for e_rem2 in sample_removed:
                    if e_rem2 == e_pres or e_rem2 == e_rem1:
                        continue
                    
                    # Check if this swap is valid
                    # Simulate: remove e_pres, add e_rem1, e_rem2
                    valid = True
                    # Check all C4s that are affected
                    affected_c4s = set()
                    for c4_idx in c4_membership[e_pres]:
                        affected_c4s.add(c4_idx)
                    for c4_idx in c4_membership[e_rem1]:
                        affected_c4s.add(c4_idx)
                    for c4_idx in c4_membership[e_rem2]:
                        affected_c4s.add(c4_idx)
                    
                    for c4_idx in affected_c4s:
                        e1, e2, e3, e4 = all_c4s[c4_idx]
                        # Count present edges in this C4 after swap
                        count = 0
                        for edge in (e1, e2, e3, e4):
                            if edge == e_pres:
                                continue  # removed
                            if edge == e_rem1 or edge == e_rem2:
                                count += 1  # added
                            elif present[edge]:
                                count += 1
                        if count == 4:
                            valid = False
                            break
                    
                    if valid:
                        # This swap is valid and gains 1 edge
                        # Prefer swaps that don't create "near-bad" C4s (3 present)
                        near_bad = 0
                        for c4_idx in affected_c4s:
                            e1, e2, e3, e4 = all_c4s[c4_idx]
                            count = 0
                            for edge in (e1, e2, e3, e4):
                                if edge == e_pres:
                                    continue
                                if edge == e_rem1 or edge == e_rem2:
                                    count += 1
                                elif present[edge]:
                                    count += 1
                            if count == 3:
                                near_bad += 1
                        
                        gain = 1 - near_bad * 0.1  # Slight penalty for near-bad C4s
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (e_pres, e_rem1, e_rem2)
        
        if best_swap is None:
            break
        
        e_pres, e_rem1, e_rem2 = best_swap
        present[e_pres] = False
        present[e_rem1] = True
        present[e_rem2] = True
        m += 1
        
        # Update C4 counts
        for c4_idx in c4_membership[e_pres]:
            c4_present_count[c4_idx] -= 1
        for c4_idx in c4_membership[e_rem1]:
            c4_present_count[c4_idx] += 1
        for c4_idx in c4_membership[e_rem2]:
            c4_present_count[c4_idx] += 1

    # 9. Final pass to add any more edges
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        progress = False
        for e in range(total_edges):
            if present[e]:
                continue
            ok = True
            for c4_idx in c4_membership[e]:
                if c4_present_count[c4_idx] >= 3:
                    ok = False
                    break
            if ok:
                present[e] = True
                m += 1
                for c4_idx in c4_membership[e]:
                    c4_present_count[c4_idx] += 1
                progress = True
        if not progress:
            break

    # 10. Construct Output
    out = []
    for idx, (u, v) in enumerate(all_edges):
        if present[idx]:
            out.append([u, v])
            
    return out

# EVOLVE-BLOCK-END

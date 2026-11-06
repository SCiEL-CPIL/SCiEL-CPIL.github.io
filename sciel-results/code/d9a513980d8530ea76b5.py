import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 12
    total_colorings = 1 << n
    all_ones = (1 << total_colorings) - 1

    # Precompute M0 and M1
    M0 = [0] * n
    M1 = [0] * n

    for c in range(total_colorings):
        for v in range(n):
            if (c >> v) & 1:
                M1[v] |= (1 << c)
            else:
                M0[v] |= (1 << c)

    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)

    # Precompute edge masks
    edge_masks = []
    for e in all_edges:
        m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
        m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
        edge_masks.append(m0 | m1)

    best_count = float('inf')
    best_edges = None

    all_indices = list(range(num_edges))
    
    # Run multiple restarts
    restarts = 200 

    for r in range(restarts):
        if time.time() > deadline - 5:
            break

        random.seed(42 + r)

        # 1. Greedy Set Cover to get initial solution
        covered = 0
        selected_indices = []
        remaining = all_indices[:]
        random.shuffle(remaining)

        while covered != all_ones and remaining:
            best_gain = -1
            best_idx = -1
            for idx in remaining:
                gain = (edge_masks[idx] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain == 0:
                break
            selected_indices.append(best_idx)
            covered |= edge_masks[best_idx]
            remaining.remove(best_idx)

        if covered != all_ones:
            continue

        # 2. Local Search with Directed Swaps and 2-for-1 Swaps
        # Maintain count of coverings for each coloring
        count = [0] * total_colorings
        for idx in selected_indices:
            m = edge_masks[idx]
            temp = m
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                count[c] += 1
                temp ^= lsb

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Deletion Phase
            indices_to_check = list(range(len(selected_indices)))
            random.shuffle(indices_to_check)
            
            deleted_any = False
            for i in indices_to_check:
                if i >= len(selected_indices):
                    continue
                e_idx = selected_indices[i]
                m = edge_masks[e_idx]
                
                can_remove = True
                temp = m
                while temp:
                    lsb = temp & -temp
                    c = lsb.bit_length() - 1
                    if count[c] == 1:
                        can_remove = False
                        break
                    temp ^= lsb
                
                if can_remove:
                    selected_indices.pop(i)
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        c = lsb.bit_length() - 1
                        count[c] -= 1
                        temp ^= lsb
                    improved = True
                    deleted_any = True
                    break

            if deleted_any:
                continue

            # Swap Phase (1-for-1)
            if selected_indices:
                e_pos = random.randint(0, len(selected_indices) - 1)
                e_idx = selected_indices[e_pos]
                m_e = edge_masks[e_idx]
                
                deficit = 0
                temp = m_e
                while temp:
                    lsb = temp & -temp
                    c = lsb.bit_length() - 1
                    if count[c] == 1:
                        deficit |= lsb
                    temp ^= lsb
                
                if deficit == 0:
                    continue

                found_swap = False
                unselected = [i for i in range(num_edges) if i not in selected_indices]
                random.shuffle(unselected)
                
                for f_idx in unselected:
                    m_f = edge_masks[f_idx]
                    if (m_f & deficit) == deficit:
                        selected_indices[e_pos] = f_idx
                        
                        temp = m_e
                        while temp:
                            lsb = temp & -temp
                            c = lsb.bit_length() - 1
                            count[c] -= 1
                            temp ^= lsb
                            
                        temp = m_f
                        while temp:
                            lsb = temp & -temp
                            c = lsb.bit_length() - 1
                            count[c] += 1
                            temp ^= lsb
                            
                        improved = True
                        found_swap = True
                        break

                if found_swap:
                    continue

            # 2-for-1 Swap Phase
            # Try to remove 2 edges and add 1 edge
            if len(selected_indices) >= 2:
                # Pick a random pair
                pair = random.sample(range(len(selected_indices)), 2)
                i1, i2 = pair
                idx1 = selected_indices[i1]
                idx2 = selected_indices[i2]
                m1 = edge_masks[idx1]
                m2 = edge_masks[idx2]
                
                # Calculate deficit if both removed
                # Deficit is set of colorings c such that count[c] - (1 if c in m1 else 0) - (1 if c in m2 else 0) == 0
                # And c is in m1 or m2 (so it was covered by them)
                # Actually, if count[c] == 1 and c in m1, removing m1 makes it 0.
                # If count[c] == 2 and c in m1 and c in m2, removing both makes it 0.
                
                deficit = 0
                # Iterate through bits in m1 | m2
                union_mask = m1 | m2
                temp = union_mask
                while temp:
                    lsb = temp & -temp
                    c = lsb.bit_length() - 1
                    # Count how many of m1, m2 cover c
                    cover_count = 0
                    if (m1 & lsb):
                        cover_count += 1
                    if (m2 & lsb):
                        cover_count += 1
                    
                    # If total count[c] becomes 0 after removal
                    if count[c] == cover_count:
                        deficit |= lsb
                    temp ^= lsb
                
                if deficit == 0:
                    # Should have been deletable individually? Or maybe both are redundant together?
                    # If deficit is 0, it means no coloring becomes uncovered.
                    # This implies we can remove both? No, we need to check if we can remove them one by one.
                    # The deletion phase would have caught single deletions.
                    # If deficit is 0 here, it means the union of their exclusive coverage is empty?
                    # If so, we can just remove one, then the other.
                    # So this case is handled by deletion.
                    continue

                # Find an edge f not in selected_indices such that mask(f) covers deficit
                unselected = [i for i in range(num_edges) if i not in selected_indices]
                random.shuffle(unselected)
                
                found_2for1 = False
                for f_idx in unselected:
                    m_f = edge_masks[f_idx]
                    if (m_f & deficit) == deficit:
                        # Valid 2-for-1 swap
                        # Remove idx1, idx2; Add f_idx
                        
                        # Update counts for idx1
                        temp = m1
                        while temp:
                            lsb = temp & -temp
                            c = lsb.bit_length() - 1
                            count[c] -= 1
                            temp ^= lsb
                            
                        # Update counts for idx2
                        temp = m2
                        while temp:
                            lsb = temp & -temp
                            c = lsb.bit_length() - 1
                            count[c] -= 1
                            temp ^= lsb
                            
                        # Update counts for f_idx
                        temp = m_f
                        while temp:
                            lsb = temp & -temp
                            c = lsb.bit_length() - 1
                            count[c] += 1
                            temp ^= lsb
                            
                        # Update selected_indices
                        # Remove i2 and i1 (be careful with indices)
                        # Since i1, i2 are indices in the list
                        # Remove the larger index first to keep the other valid
                        if i1 > i2:
                            selected_indices.pop(i1)
                            selected_indices.pop(i2)
                        else:
                            selected_indices.pop(i2)
                            selected_indices.pop(i1)
                        
                        selected_indices.append(f_idx)
                        
                        improved = True
                        found_2for1 = True
                        break

                if found_2for1:
                    continue

        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_edges = [list(all_edges[i]) for i in selected_indices]
            
            if best_count < 51:
                break

    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
    else:
        best_n = n

    return {'n': best_n, 'edges': best_edges}

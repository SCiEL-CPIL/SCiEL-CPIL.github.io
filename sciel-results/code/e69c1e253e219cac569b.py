<<<<<<< SEARCH
def _calc_delta_energy(sequence, chain, k, new_segment, occupied, is_h):
    """
    Calculate the change in energy when replacing segment k: of chain with new_segment.
    'occupied' must contain the rest of the chain (0..k-1) and NOT the old segment k..n-1.
    """
    n = len(chain)
    old_seg = chain[k:]
    
    # 1. Count contacts lost by removing old_seg
    # Contacts are between H residues at distance 1, not adjacent in chain.
    # We need to count:
    # a) Contacts between old_seg and rest (0..k-1)
    # b) Contacts within old_seg (non-adjacent)
    
    lost = 0
    
    # Create a set of positions in old_seg for fast internal contact check?
    # Or just iterate.
    old_pos_set = set(old_seg)
    
    for i_offset, pos in enumerate(old_seg):
        idx = k + i_offset
        if not is_h[idx]:
            continue
        for nb in _neighbours(pos):
            # Check if neighbor is in rest (0..k-1)
            if nb in occupied:
                nb_idx = occupied[nb]
                # nb_idx is in 0..k-1
                if is_h[nb_idx]:
                    # Check if adjacent in chain?
                    # idx is k..n-1, nb_idx is 0..k-1. 
                    # Adjacent means |idx - nb_idx| == 1.
                    # Since nb_idx < k and idx >= k, |idx - nb_idx| == 1 is possible if idx==k and nb_idx==k-1.
                    if abs(idx - nb_idx) > 1:
                        lost += 1
            
            # Check if neighbor is in old_seg (internal contact)
            elif nb in old_pos_set:
                # Find index of nb in old_seg
                # We can't use 'occupied' for this part as it doesn't have old_seg.
                # We need to know the index in the chain.
                # Let's pre-compute a map for old_seg positions to their chain indices?
                # Or just check distance.
                # Since we are iterating over old_seg, we can just check if nb is in old_seg.
                # But we need to know if they are adjacent in chain.
                # Let's build a small dict for old_seg: pos -> idx
                pass # This is getting inefficient. Let's use a different approach for delta.

    # Better Delta Approach:
    # 1. Calculate energy of the "rest" + "old_seg" ? No.
    # 2. Calculate contacts involving the boundary and internal to the segment.
    
    # Let's just re-calculate the contacts involving the moved segment.
    # Total Energy E = -Contacts
    # Delta E = E_new - E_old
    # Contacts_old = C(rest, rest) + C(rest, old_seg) + C(old_seg, old_seg)
    # Contacts_new = C(rest, rest) + C(rest, new_seg) + C(new_seg, new_seg)
    # Delta Contacts = [C(rest, new_seg) + C(new_seg, new_seg)] - [C(rest, old_seg) + C(old_seg, old_seg)]
    
    # C(A, B) counts H-H pairs where one is in A and one is in B, and they are neighbors.
    # If A and B are disjoint sets of positions, and we check neighbors, we count each pair once if we iterate A and check B.
    # For internal C(old_seg, old_seg), we must avoid double counting.
    
    def count_contacts_between(seg_positions, seg_start_idx, other_positions_set, other_indices_dict):
        """
        Count H-H contacts between seg (indices seg_start_idx...) and other (positions in other_positions_set).
        other_indices_dict maps position to index in chain.
        """
        c = 0
        for i_offset, pos in enumerate(seg_positions):
            idx = seg_start_idx + i_offset
            if not is_h[idx]:
                continue
            for nb in _neighbours(pos):
                if nb in other_positions_set:
                    nb_idx = other_indices_dict[nb]
                    if is_h[nb_idx]:
                        if abs(idx - nb_idx) > 1:
                            c += 1
        return c

    # 1. C(rest, old_seg)
    # rest is in 'occupied' (pos -> idx)
    # old_seg is chain[k:]
    c_rest_old = count_contacts_between(old_seg, k, set(occupied.keys()), occupied)
    
    # 2. C(old_seg, old_seg)
    # We need to count contacts within old_seg.
    # To avoid double counting, we can count pairs (i, j) with i < j and |i-j|>1.
    # Or use the neighbor check but divide by 2?
    # Let's use a set of old_seg positions for lookup.
    old_pos_to_idx = {pos: k + i for i, pos in enumerate(old_seg)}
    
    c_old_internal = 0
    for i_offset, pos in enumerate(old_seg):
        idx = k + i_offset
        if not is_h[idx]:
            continue
        for nb in _neighbours(pos):
            if nb in old_pos_to_idx:
                nb_idx = old_pos_to_idx[nb]
                if is_h[nb_idx]:
                    # Count each pair once. Let's say we only count if idx < nb_idx.
                    if idx < nb_idx and abs(idx - nb_idx) > 1:
                        c_old_internal += 1
    
    # 3. C(rest, new_seg)
    c_rest_new = count_contacts_between(new_segment, k, set(occupied.keys()), occupied)
    
    # 4. C(new_seg, new_seg)
    new_pos_to_idx = {pos: k + i for i, pos in enumerate(new_segment)}
    c_new_internal = 0
    for i_offset, pos in enumerate(new_segment):
        idx = k + i_offset
        if not is_h[idx]:
            continue
        for nb in _neighbours(pos):
            if nb in new_pos_to_idx:
                nb_idx = new_pos_to_idx[nb]
                if is_h[nb_idx]:
                    if idx < nb_idx and abs(idx - nb_idx) > 1:
                        c_new_internal += 1
                        
    delta_contacts = (c_rest_new + c_new_internal) - (c_rest_old + c_old_internal)
    delta_energy = -delta_contacts
    return delta_energy
=======
def _calc_delta_energy(sequence, chain, k, new_segment, occupied, occupied_set, is_h):
    """
    Calculate the change in energy when replacing segment k: of chain with new_segment.
    'occupied' must contain the rest of the chain (0..k-1) and NOT the old segment k..n-1.
    'occupied_set' is the set of keys in 'occupied'.
    """
    n = len(chain)
    old_seg = chain[k:]
    
    # Helper to count contacts between a segment and the rest of the chain
    def count_contacts_with_rest(seg_positions, seg_start_idx):
        c = 0
        for i_offset, pos in enumerate(seg_positions):
            idx = seg_start_idx + i_offset
            if not is_h[idx]:
                continue
            for nb in _neighbours(pos):
                if nb in occupied_set:
                    nb_idx = occupied[nb]
                    if is_h[nb_idx]:
                        if abs(idx - nb_idx) > 1:
                            c += 1
        return c

    # Helper to count internal contacts within a segment
    def count_internal_contacts(seg_positions, seg_start_idx):
        # Build a local map for fast lookup
        pos_to_idx = {pos: seg_start_idx + i for i, pos in enumerate(seg_positions)}
        c = 0
        for i_offset, pos in enumerate(seg_positions):
            idx = seg_start_idx + i_offset
            if not is_h[idx]:
                continue
            for nb in _neighbours(pos):
                if nb in pos_to_idx:
                    nb_idx = pos_to_idx[nb]
                    if is_h[nb_idx]:
                        if idx < nb_idx and abs(idx - nb_idx) > 1:
                            c += 1
        return c

    c_rest_old = count_contacts_with_rest(old_seg, k)
    c_old_internal = count_internal_contacts(old_seg, k)
    
    c_rest_new = count_contacts_with_rest(new_segment, k)
    c_new_internal = count_internal_contacts(new_segment, k)
    
    delta_contacts = (c_rest_new + c_new_internal) - (c_rest_old + c_old_internal)
    delta_energy = -delta_contacts
    return delta_energy
>>>>>>> REPLACE


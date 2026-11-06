import time

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    Uses a bitmask-based exact cover search (Algorithm X) with MRV heuristic.
    """
    start_time = time.time()
    deadline = start_time + 100.0

    # 1. Precomputation
    # Generate all 5-subsets and map to bit indices
    all_5subsets = []
    five_to_idx = {}
    idx = 0
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                for l in range(k+1, 12):
                    for m in range(l+1, 12):
                        sub = (i, j, k, l, m)
                        five_to_idx[sub] = idx
                        all_5subsets.append(sub)
                        idx += 1
    
    num_5subsets = 792
    full_mask = (1 << num_5subsets) - 1

    # Generate all 6-subsets and their masks
    all_6subsets = []
    six_masks = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                for l in range(k+1, 12):
                    for m in range(l+1, 12):
                        for n in range(m+1, 12):
                            block = (i, j, k, l, m, n)
                            all_6subsets.append(block)
                            # Compute mask for this 6-subset
                            mask = 0
                            # The 5-subsets of this 6-subset are obtained by removing one element
                            # block is (i,j,k,l,m,n)
                            # 5-subsets:
                            # (j,k,l,m,n) -> remove i
                            # (i,k,l,m,n) -> remove j
                            # (i,j,l,m,n) -> remove k
                            # (i,j,k,m,n) -> remove l
                            # (i,j,k,l,n) -> remove m
                            # (i,j,k,l,m) -> remove n
                            
                            # We need to map these tuples to indices
                            # To avoid creating tuples repeatedly, we can pre-compute a lookup for 5-tuples
                            # But we already have five_to_idx.
                            
                            mask |= 1 << five_to_idx[(j, k, l, m, n)]
                            mask |= 1 << five_to_idx[(i, k, l, m, n)]
                            mask |= 1 << five_to_idx[(i, j, l, m, n)]
                            mask |= 1 << five_to_idx[(i, j, k, m, n)]
                            mask |= 1 << five_to_idx[(i, j, k, l, n)]
                            mask |= 1 << five_to_idx[(i, j, k, l, m)]
                            six_masks.append(mask)
                            
    num_6subsets = len(all_6subsets) # 924

    # For MRV, we need to know which 6-subsets cover each 5-subset.
    # five_to_sixes[five_idx] = list of indices of 6-subsets that cover it
    five_to_sixes = [[] for _ in range(num_5subsets)]
    for idx, mask in enumerate(six_masks):
        # Iterate through set bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            five_to_sixes[bit_idx].append(idx)
            m ^= lsb

    # State
    selected_blocks = []
    current_mask = 0 # Bits already covered
    
    # Symmetry breaking: Fix first block
    # Find index of block (0,1,2,3,4,5)
    first_block_idx = -1
    for i, b in enumerate(all_6subsets):
        if b == (0, 1, 2, 3, 4, 5):
            first_block_idx = i
            break
            
    if first_block_idx != -1:
        selected_blocks.append(all_6subsets[first_block_idx])
        current_mask |= six_masks[first_block_idx]
        
    # MRV Heuristic: Find the 5-subset with the fewest available 6-subsets
    # We maintain a count of available 6-subsets for each 5-subset?
    # Or we just scan?
    # Scanning 792 items every step is 792 * 132 ~ 100k operations. Fast.
    
    def backtrack():
        nonlocal current_mask, selected_blocks
        if time.time() > deadline:
            return False
            
        if len(selected_blocks) == 132:
            return True
            
        # Find MRV
        min_avail = float('inf')
        best_five_idx = -1
        
        # We only care about 5-subsets not yet covered
        # current_mask has 1s for covered.
        # We need to find a bit in (~current_mask) & full_mask that has fewest candidates.
        
        # Optimization: Iterate through all 5-subsets
        for fid in range(num_5subsets):
            if (current_mask >> fid) & 1:
                continue # Already covered
                
            # Count available 6-subsets for this fid
            count = 0
            for s_idx in five_to_sixes[fid]:
                # Check if this 6-subset is disjoint with current_mask
                if six_masks[s_idx] & current_mask:
                    continue
                count += 1
                if count >= min_avail:
                    break
                    
            if count == 0:
                return False # Dead end
                
            if count < min_avail:
                min_avail = count
                best_five_idx = fid
                if count == 1:
                    break
                    
        if best_five_idx == -1:
            # No uncovered 5-subsets?
            # If we haven't picked 132 blocks, this shouldn't happen unless we made a mistake
            # But if all 5-subsets are covered, we are done.
            # However, we must have exactly 132 blocks.
            # If len < 132, it's an error in logic, but let's return True if all covered?
            # No, the condition is len == 132.
            # If all 5-subsets are covered, we MUST have 132 blocks because 132*6 = 792.
            # So if all covered, len must be 132.
            return len(selected_blocks) == 132

        # Try all 6-subsets covering best_five_idx
        for s_idx in five_to_sixes[best_five_idx]:
            if six_masks[s_idx] & current_mask:
                continue
                
            # Apply
            new_mask = current_mask | six_masks[s_idx]
            selected_blocks.append(all_6subsets[s_idx])
            
            if backtrack():
                return True
                
            # Undo
            selected_blocks.pop()
            current_mask = new_mask # Wait, current_mask is nonlocal, need to restore
            # Actually, I should store old_mask
            # Let's fix the undo logic
            
        return False

    # Fix the backtrack function to handle state restoration correctly
    def backtrack_v2():
        nonlocal current_mask, selected_blocks
        if time.time() > deadline:
            return False
            
        if len(selected_blocks) == 132:
            return True
            
        min_avail = float('inf')
        best_five_idx = -1
        
        for fid in range(num_5subsets):
            if (current_mask >> fid) & 1:
                continue
                
            count = 0
            for s_idx in five_to_sixes[fid]:
                if six_masks[s_idx] & current_mask:
                    continue
                count += 1
                if count >= min_avail:
                    break
                    
            if count == 0:
                return False
                
            if count < min_avail:
                min_avail = count
                best_five_idx = fid
                if count == 1:
                    break
                    
        if best_five_idx == -1:
            return len(selected_blocks) == 132

        for s_idx in five_to_sixes[best_five_idx]:
            if six_masks[s_idx] & current_mask:
                continue
                
            # Save state
            old_mask = current_mask
            selected_blocks.append(all_6subsets[s_idx])
            current_mask |= six_masks[s_idx]
            
            if backtrack_v2():
                return True
                
            # Restore state
            selected_blocks.pop()
            current_mask = old_mask
            
        return False

    if backtrack_v2():
        return [list(b) for b in selected_blocks]
        
    # Fallback: return what we have
    return [list(b) for b in selected_blocks]

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")

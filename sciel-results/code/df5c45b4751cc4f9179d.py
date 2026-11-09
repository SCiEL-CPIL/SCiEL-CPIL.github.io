<<<<<<< SEARCH
    # Local search with bitset operations
    def local_search_bitset(mask, max_iters=5000):
        mask = mask
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # Try adding each non-chosen node
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
            
            # Try swapping: for each non-chosen node i, try removing one conflict
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    # Can add directly (should have been caught above, but just in case)
                    mask |= (1 << i)
                    improved = True
                    continue
                # Try removing each bit in conf_in_mask
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    # Check if i conflicts with any other node in mask besides r_idx
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit
        
        return mask
=======
    # Local search with bitset operations - improved with multi-swap
    def local_search_bitset(mask, max_iters=5000):
        mask = mask
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # Try adding each non-chosen node
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
            
            # Try 1-for-1 swaps: for each non-chosen node i, try removing one conflict
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    mask |= (1 << i)
                    improved = True
                    continue
                # Try removing each bit in conf_in_mask
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit
            
            # Try 2-for-1 swaps: remove two nodes, add one
            if iters % 5 == 0:  # Do this less frequently
                for i in range(nv):
                    if mask & (1 << i):
                        continue
                    conf_in_mask = conflicts[i] & mask
                    if conf_in_mask == 0:
                        mask |= (1 << i)
                        improved = True
                        continue
                    # Get list of conflicting nodes
                    conf_list = []
                    temp = conf_in_mask
                    while temp:
                        r_bit = temp & (-temp)
                        r_idx = r_bit.bit_length() - 1
                        conf_list.append(r_idx)
                        temp ^= r_bit
                    
                    # Try removing pairs of conflicts
                    for a_idx in range(len(conf_list)):
                        for b_idx in range(a_idx + 1, len(conf_list)):
                            a = conf_list[a_idx]
                            b = conf_list[b_idx]
                            a_bit = 1 << a
                            b_bit = 1 << b
                            new_mask = mask & ~a_bit & ~b_bit
                            if (conflicts[i] & new_mask) == 0:
                                mask = new_mask | (1 << i)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return mask
>>>>>>> REPLACE


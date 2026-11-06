    for i, S in enumerate(all_8_subsets):
        # Build 64-bit mask for edges in S
        m = 0
        for k in range(8):
            for l in range(k+1, 8):
                if (adj[S[k]] >> S[l]) & 1:
                    m |= (1 << (k*8 + l))
                    m |= (1 << (l*8 + k)) # Or just check one direction? 
                    # If we store both, the mask check needs to account for it.
                    # Simpler: store only i<j bits. 28 bits.
        # Let's stick to i<j bits. Bit pos = k*7 + (l-1-k) is messy.
        # Bit pos = k*8 + l is sparse but easy. Max bit 7*8+7 = 63. Fits in 64-bit.
        
        for mask in q3_masks[i]:
            if (m & mask) == mask:
                return False
    return True
    
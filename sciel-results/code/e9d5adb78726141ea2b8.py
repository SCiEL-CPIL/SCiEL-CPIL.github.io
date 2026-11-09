import time

def construct():
    p = 1009
    m = 5
    deadline = time.time() + 85.0
    
    # Precompute roots and build the bitset of non-zero quadratic residues
    # root_map[r] = smallest x in [1, p-1] such that x^2 % p == r
    root_map = {}
    residue_mask = 0  # Bit i is 1 if i is a non-zero QR
    
    for x in range(1, p):
        if time.time() > deadline:
            # Fallback if precomputation is too slow (unlikely)
            # Just return a known valid late run or raise
            # For p=1009, this loop is extremely fast
            raise TimeoutError("Computation exceeded deadline")
            
        r = (x * x) % p
        if r == 0:
            continue
        
        # Set the bit for this residue
        residue_mask |= (1 << r)
        
        # Store the smallest root
        if r not in root_map:
            root_map[r] = x
    
    # We need s such that s, s+1, ..., s+4 are all QRs.
    # This is equivalent to finding the smallest s >= 1 such that
    # bit s, s+1, s+2, s+3, s+4 are all set in residue_mask.
    
    # Let V be the mask of valid start positions.
    # V has bit s set if residue_mask has bits s, s+1, s+2, s+3, s+4 set.
    # V = residue_mask & (residue_mask >> 1) & (residue_mask >> 2) & (residue_mask >> 3) & (residue_mask >> 4)
    
    v = residue_mask
    for i in range(1, m):
        v &= (residue_mask >> i)
        
    # We need the smallest s >= 1 that is a set bit in v.
    # Clear bit 0 (value 0) because s must be >= 1.
    v &= ~1
    
    if v == 0:
        # Fallback: search linearly if bitwise logic failed or no run found
        # This should not happen for p=1009, m=5
        for s in range(1, p - m + 1):
            if all((s+i) in root_map for i in range(m)):
                return {'start': s, 'roots': [root_map[s+i] for i in range(m)]}
        raise RuntimeError("No run of 5 consecutive nonzero quadratic residues found")
        
    # Find the index of the lowest set bit in v
    # lowest_set_bit = v & -v
    # s = lowest_set_bit.bit_length() - 1
    
    lowest_set_bit = v & -v
    s = lowest_set_bit.bit_length() - 1
    
    # Verify s is within valid range [1, 1004]
    if s < 1 or s > p - m:
        # If the lowest bit is out of range (e.g., if we had wrapped around issues, 
        # but here bits correspond directly to values 1..1008, and shifts handle the tail),
        # we might need to check the next bits. 
        # However, since we cleared bit 0, and the max index in residue_mask is 1008,
        # the max valid s is 1004. 
        # If s > 1004, it means the only runs found start too late? 
        # But residue_mask only has bits up to 1008.
        # If s=1005, bits 1005..1009 would need to be set. 
        # Bit 1009 is 0 (since max residue is 1008). 
        # So s cannot be > 1004 if the bits are derived correctly from 1..1008.
        # Just in case, iterate up.
        for i in range(s, p - m + 1):
            if (v >> i) & 1:
                s = i
                break
        else:
            raise RuntimeError("No valid start found in range")
            
    roots = [root_map[s + i] for i in range(m)]
    
    return {'start': s, 'roots': roots}

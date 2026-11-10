"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    
    # Bitset for quadratic residues
    # Bit i is 1 if i is a non-zero quadratic residue mod p
    mask = 0
    # roots[i] stores a root r such that r^2 % p == i, or 0 if i is not a QR
    roots = [0] * p
    
    for x in range(1, p):
        r2 = (x * x) % p
        # Set the bit for r2
        mask |= (1 << r2)
        # Store the root. We keep the first root found (smallest x)
        if roots[r2] == 0:
            roots[r2] = x
    
    # valid_mask bit s is 1 if s, s+1, s+2, s+3, s+4 are all QRs
    # mask >> k has bit s set if mask had bit s+k set
    valid_mask = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    
    # Find the least significant set bit in valid_mask
    if valid_mask == 0:
        # Fallback: linear search (should not happen)
        for s in range(1, p - m + 1):
            if all((s + i) < p and roots[s + i] != 0 for i in range(m)):
                return {'start': s, 'roots': [roots[s + i] for i in range(m)]}
        raise RuntimeError('No run found')
        
    # Isolate the lowest set bit
    lsb = valid_mask & -valid_mask
    s = lsb.bit_length() - 1
    
    # Verify constraints: 1 <= s <= 1004
    # Since bit 0 of mask is 0 (0 is not a non-zero QR), bit 0 of valid_mask is 0.
    # So s >= 1.
    # Since mask has 0s for indices >= p, and we shift right, valid_mask will be 0
    # for s such that s+4 >= p. i.e., s >= 1005.
    # So s <= 1004 is guaranteed if s is a valid index from the bitset.
    
    if s < 1 or s > 1004:
         # This case should theoretically not occur given the bitset logic, 
         # but just in case, do a linear check
         for s_check in range(1, p - m + 1):
             if all(roots[s_check + i] != 0 for i in range(m)):
                 s = s_check
                 break
         else:
             raise RuntimeError('No valid s found in range')

    result_roots = [roots[s + i] for i in range(m)]
    
    return {'start': s, 'roots': result_roots}
# EVOLVE-BLOCK-END

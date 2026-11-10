import time

def construct():
    start_time = time.time()
    deadline = 80.0
    
    p = 1009
    
    # root_of[i] stores a root r such that r*r % p == i
    root_of = [-1] * p
    
    # Build a bitset where bit i is set if i is a quadratic residue mod p
    # We only care about residues 1..1008 (non-zero)
    qr_mask = 0
    for r in range(1, p):
        res = (r * r) % p
        if res > 0:  # exclude zero residue
            if root_of[res] == -1:
                root_of[res] = r
            qr_mask |= (1 << res)
    
    # Use bitwise operations to find all positions s where s, s+1, s+2, s+3, s+4 are all residues
    # valid_mask has bit s set if s through s+4 are all quadratic residues
    valid_mask = qr_mask & (qr_mask >> 1) & (qr_mask >> 2) & (qr_mask >> 3) & (qr_mask >> 4)
    
    # We need s in range [1, 1004] so that s+4 <= 1008
    # Mask out positions outside [1, 1004]
    # Bit positions 1 through 1004 are valid
    valid_range_mask = 0
    for i in range(1, 1005):
        valid_range_mask |= (1 << i)
    
    valid_mask &= valid_range_mask
    
    # Find the smallest set bit (smallest s)
    if valid_mask:
        # Find the least significant set bit
        s = (valid_mask & (-valid_mask)).bit_length() - 1
        return {
            'start': s,
            'roots': [root_of[s], root_of[s+1], root_of[s+2], root_of[s+3], root_of[s+4]]
        }
    
    # Fallback if not found (should not happen for p=1009)
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}

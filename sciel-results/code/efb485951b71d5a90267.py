"""Find minimum start s for 5 consecutive quadratic residues mod 1009."""
import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Step 1: Build a bitset of which residues are quadratic residues (non-zero)
    # residue_mask[i] = True if i is a non-zero QR mod 1009, for i in 1..1008
    # We'll use a Python integer as a bitset, or a bytearray for clarity.
    # Actually, let's use a set for roots and a boolean array for membership.
    
    # Precompute roots: for each non-zero QR, store one root
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots:
                roots[r] = x
    
    # Create a boolean array: is_qr[i] = True if i is in roots (i.e., i is a non-zero QR)
    # Index 0 is not a valid residue (zero excluded), indices 1..1008
    is_qr = [False] * (p + 1)
    for r in roots:
        is_qr[r] = True
    
    # Step 2: Use bitset AND over shifted masks to find all valid 5-runs in one pass
    # We want to find the minimum s in [1, 1004] such that s, s+1, s+2, s+3, s+4 are all QRs.
    # 
    # Approach: Create a bitset where bit i is set if i is a QR.
    # Then for a run starting at s, we need bits s, s+1, s+2, s+3, s+4 all set.
    # This is equivalent to: (bitset >> s) & 0b11111 == 0b11111
    # 
    # More efficiently: 
    # Let B be the bitset where bit i is set if i is a QR (for i in 1..1008).
    # We want positions s where B[s..s+4] are all 1.
    # We can compute: B & (B >> 1) & (B >> 2) & (B >> 3) & (B >> 4)
    # The bit at position s in this AND result is 1 iff s, s+1, s+2, s+3, s+4 are all QRs.
    # Then we find the least set bit in the range [1, 1004].
    
    # Build the bitset as a Python int. Bit i corresponds to residue i.
    # We need bits 1 through 1008 to be meaningful.
    bitset = 0
    for r in range(1, p):
        if is_qr[r]:
            bitset |= (1 << r)
    
    # Compute the AND of shifted versions
    # (B >> k) has bit i set iff B had bit (i+k) set.
    # So B & (B>>1) & (B>>2) & (B>>3) & (B>>4) has bit s set iff B[s], B[s+1], ..., B[s+4] are all set.
    mask = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    # Now find the least s in [1, 1004] such that bit s is set in mask.
    # We can iterate s from 1 to 1004 and check the bit.
    best_s = None
    for s in range(1, 1005):
        if (mask >> s) & 1:
            best_s = s
            break
    
    if best_s is None:
        # Fallback: linear scan (should not happen given the problem structure)
        for s in range(1, 1005):
            if all(is_qr[s + i] for i in range(5)):
                best_s = s
                break
    
    if best_s is None:
        raise RuntimeError('No valid run found')
    
    # Extract roots
    result_roots = [roots[best_s + i] for i in range(5)]
    
    return {'start': best_s, 'roots': result_roots}

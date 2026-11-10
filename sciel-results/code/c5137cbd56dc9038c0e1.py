"""Rewritten program using bitwise AND of shifted residue bitsets for O(1) minimal-start search."""
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s wall-clock limit
    p = 1009
    m = 5
    
    # Build a bitset where bit r is set iff r is a quadratic residue mod p (r != 0)
    # and r is in range 1..p-1
    # We need to check if s, s+1, ..., s+4 are all quadratic residues (non-zero)
    # s ranges from 1 to p-1-m = 1004
    # s+4 ranges from 5 to 1008, all within 1..1008, so no wraparound issues
    
    # Create bitset as a Python int for arbitrary precision
    # bit i set means residue i is a quadratic residue (and non-zero)
    residue_bits = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_bits |= (1 << r)
    
    # Use bitwise AND of shifted residue bitsets to find all valid start positions simultaneously
    # mask = bits where s, s+1, s+2, s+3, s+4 are all set
    mask = residue_bits & (residue_bits >> 1) & (residue_bits >> 2) & (residue_bits >> 3) & (residue_bits >> 4)
    
    # Find the smallest s in [1, 1004] such that bit s is set in mask
    # This is equivalent to finding the lowest set bit in mask that is >= 1 and <= 1004
    # We need to mask out bits below 1 and above 1004
    valid_mask = mask & 0x1FFFE  # bits 1 through 1004 (0x1FFFE = 2^14 - 2)
    
    if valid_mask == 0:
        raise RuntimeError('No baseline run found')
    
    # Find the lowest set bit position (0-indexed)
    # (valid_mask & -valid_mask) gives the lowest set bit
    # bit_length() gives the position + 1
    lowest_bit = valid_mask & (-valid_mask)
    s = lowest_bit.bit_length() - 1
    
    # Verify s is in valid range
    if s < 1 or s > 1004:
        raise RuntimeError(f'Invalid start position: {s}')
    
    # Find roots for s, s+1, ..., s+4
    roots = []
    for i in range(m):
        target = s + i
        # Find a root x in [1, 1008] such that x^2 mod p == target
        found = False
        for x in range(1, p):
            if (x * x) % p == target:
                roots.append(x)
                found = True
                break
        if not found:
            raise RuntimeError(f"No root found for residue {target}")
    
    return {'start': s, 'roots': roots}

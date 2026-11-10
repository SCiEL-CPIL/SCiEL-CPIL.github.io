"""Rewritten program using bitset approach to find earliest run of 5 consecutive quadratic residues."""
import time

def construct():
    # Deadline guard: return best found before 80s (safety margin under 90s)
    start_time = time.time()
    deadline = start_time + 80.0
    
    p = 1009
    
    # Build a bitset where bit i is set if i is a quadratic residue mod p (i in 1..p-1)
    # Also store one root for each residue
    roots = [0] * p  # roots[r] = x such that x*x % p == r, 0 if not set
    is_residue = [False] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r > 0 and not is_residue[r]:
            is_residue[r] = True
            roots[r] = x
    
    # Build a bitset as a Python integer for fast bitwise operations
    # Bit i (0-indexed) corresponds to residue i
    bitset = 0
    for i in range(1, p):
        if is_residue[i]:
            bitset |= (1 << i)
    
    # We need to find the smallest s (1 <= s <= 1004) such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues.
    # Using bitset: shift the bitset right by i for i=0..4 and AND them together.
    # A bit at position s in the AND result means s, s+1, s+2, s+3, s+4 are all residues.
    
    # AND all 5 shifted versions
    combined = bitset
    for shift in range(1, 5):
        combined &= (bitset >> shift)
    
    # Now find the lowest set bit in combined (which corresponds to the smallest s)
    # We need s >= 1 and s <= 1004
    # The lowest set bit position gives us s
    if combined == 0:
        raise RuntimeError('No run of 5 consecutive quadratic residues found')
    
    # Find the lowest set bit
    # (combined & -combined) isolates the lowest set bit
    lowest = combined & (-combined)
    s = lowest.bit_length() - 1
    
    # Verify s is in valid range
    if s < 1 or s > 1004:
        # Find next valid s by scanning
        # Create a mask for valid positions 1..1004
        valid_mask = 0
        for i in range(1, 1005):
            valid_mask |= (1 << i)
        combined_valid = combined & valid_mask
        if combined_valid == 0:
            raise RuntimeError('No valid run found')
        lowest = combined_valid & (-combined_valid)
        s = lowest.bit_length() - 1
    
    # Build the roots list
    result_roots = []
    for i in range(5):
        r = s + i
        result_roots.append(roots[r])
    
    # Verify correctness
    for i in range(5):
        assert (result_roots[i] ** 2) % p == s + i, f"Verification failed for s={s}, i={i}"
    
    return {'start': s, 'roots': result_roots}

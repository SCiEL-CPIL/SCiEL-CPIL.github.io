"""Find the smallest start s such that s, s+1, s+2, s+3, s+4 are all quadratic residues mod 1009."""
import time

def construct():
    start_time = time.time()
    p = 1009
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Precompute: for each quadratic residue r in 1..p-1, store one root x
    # such that x*x % p == r. Also build a bitmask of which values are residues.
    roots = [0] * p  # roots[r] = x if r is a QR, else 0
    is_residue = [False] * p

    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and not is_residue[r]:
            is_residue[r] = True
            roots[r] = x

    # Use bitwise AND of shifted residue indicators to find minimal s in O(1)
    # after O(p) setup. Build a bitset where bit i is set iff i is a QR.
    # Then check consecutive runs via shifting and ANDing.

    # Build a Python int as a bitset: bit i set means i is a quadratic residue (1..1008)
    bitset = 0
    for i in range(1, p):
        if is_residue[i]:
            bitset |= (1 << i)

    # We need s, s+1, s+2, s+3, s+4 all to be residues.
    # Equivalent: (bitset >> s) & 0b11111 == 0b11111
    # Search s from 1 to 1004.
    # But we can be smarter: find the first set of 5 consecutive 1-bits starting at position >=1 and ending <= 1008.

    # Shift bitset left by 1 to discard bit 0 (we only care about positions 1..1008)
    # Actually, let's just iterate but use the bit trick for speed.
    # For each s, check if bits s through s+4 are all set.

    # More efficient: compute the AND of bitset shifted by 0,1,2,3,4 positions.
    # A position s has all 5 bits set iff (bitset & (bitset>>1) & (bitset>>2) & (bitset>>3) & (bitset>>4)) has bit s set.
    # But we need bits s, s+1, s+2, s+3, s+4 set in the original.
    # That means: bit s in bitset, bit s in (bitset>>1) [which is bit s+1 in original], etc.
    # So: combined = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    # Bit s in combined is set iff bits s, s+1, s+2, s+3, s+4 are all set in bitset.

    combined = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)

    # Now find the smallest s in 1..1004 such that bit s is set in combined.
    # Bit s of combined: (combined >> s) & 1
    # We want the lowest such s >= 1.

    # Mask out bit 0 (s=0 is not valid, we need s>=1)
    combined &= ~1  # clear bit 0

    # Also, s+4 <= 1008, so s <= 1004. Bits above 1004 don't matter for s, but
    # if bit s is set in combined, it means bits s..s+4 are set. Since max bit in bitset is 1008,
    # bit 1005 in combined would require bits 1005..1009, but 1009 doesn't exist (max is 1008).
    # So bits 1005+ in combined are automatically 0. Good.

    # Find the lowest set bit in combined (that's >= bit 1)
    if combined == 0:
        raise RuntimeError('No valid run found')

    # lowest set bit: combined & (-combined) gives the lowest set bit as a power of 2
    lowest = combined & (-combined)
    s = lowest.bit_length() - 1

    # s should be in 1..1004
    assert 1 <= s <= 1004, f"Unexpected s={s}"

    result_roots = [roots[s + i] for i in range(5)]

    return {'start': s, 'roots': result_roots}

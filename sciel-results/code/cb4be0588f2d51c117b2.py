"""Find the minimum start s for 5 consecutive quadratic residues mod 1009 using bitset."""
import time

def construct():
    p = 1009
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Build bitset of non-zero quadratic residues
    # bit i is 1 if i is a non-zero quadratic residue mod p
    bits = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            bits |= (1 << r)
    
    # We need to find minimum s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues (non-zero)
    # and s+4 <= 1008 (since residues must be <= 1008, i.e., non-zero mod 1009)
    # Wait: the constraint says 1<=ri<=1008, and ri^2 mod 1009 == s+i
    # So s+i must be a non-zero quadratic residue, meaning 1 <= s+i <= 1008
    # Thus s >= 1 and s+4 <= 1008, so s <= 1004. That's consistent.
    
    # Using bitset: create a mask where bit j is 1 if j is a QR.
    # For a given s, we need bits at positions s, s+1, s+2, s+3, s+4 all set.
    # We can create shifted versions and AND them.
    
    # mask_s = bits & (bits >> 1) & (bits >> 2) & (bits >> 3) & (bits >> 4)
    # But careful: we need bit s in bits, bit s in (bits>>1) means bit s+1 in bits, etc.
    # Actually: (bits >> 1) has bit j set if bits had bit j+1 set.
    # So (bits >> 1) at position s checks if s+1 is a QR.
    # So mask = bits & (bits >> 1) & (bits >> 2) & (bits >> 3) & (bits >> 4)
    # Bit s in mask is 1 iff s, s+1, s+2, s+3, s+4 are all QRs.
    
    mask = bits & (bits >> 1) & (bits >> 2) & (bits >> 3) & (bits >> 4)
    
    # Find the lowest set bit in mask at position s where 1 <= s <= 1004
    # We need s such that bit s is set in mask, and s >= 1, s <= 1004.
    # Also s+4 <= 1008 means s <= 1004. And s >= 1.
    
    # Mask out positions 0 and > 1004
    # We want bits 1 through 1004 inclusive.
    valid_mask = mask & 0
    for i in range(1, 1005):
        valid_mask |= (1 << i)
    valid_mask &= mask
    
    if valid_mask == 0:
        # Fallback: linear search
        for s in range(1, 1005):
            if all(((s + i) & 0) and ((bits >> (s + i)) & 1) for i in range(5)):
                roots = []
                for i in range(5):
                    r = s + i
                    # Find a root
                    for x in range(1, p):
                        if (x * x) % p == r:
                            roots.append(x)
                            break
                return {'start': s, 'roots': roots}
        raise RuntimeError('No run found')
    
    # Find lowest set bit
    # valid_mask & (-valid_mask) gives lowest set bit
    lowest = valid_mask & (-valid_mask)
    s = lowest.bit_length() - 1
    
    # Verify
    if s < 1 or s > 1004:
        raise RuntimeError(f'Invalid s={s}')
    
    # Find roots for s, s+1, ..., s+4
    roots = []
    for i in range(5):
        r = s + i
        # Find a root x such that x^2 mod p == r
        # We can precompute or search
        found = False
        for x in range(1, p):
            if (x * x) % p == r:
                roots.append(x)
                found = True
                break
        if not found:
            raise RuntimeError(f'No root for residue {r}')
    
    return {'start': s, 'roots': roots}

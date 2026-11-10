"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""

def construct():
    p = 1009
    m = 5
    
    # Precompute quadratic residues
    # is_qr[v] is True if v is a non-zero quadratic residue mod p
    is_qr = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        r2 = (x * x) % p
        if not is_qr[r2]:
            is_qr[r2] = True
            roots[r2] = x
        
    # Search for minimum s
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+4 are all QRs
        # Since s <= 1004 and m=5, s+4 <= 1008, so no wraparound to 0
        valid = True
        for i in range(m):
            if not is_qr[s + i]:
                valid = False
                break
        
        if valid:
            return {'start': s, 'roots': [roots[s + i] for i in range(m)]}
            
    # Fallback (should not happen)
    raise RuntimeError('No run found')

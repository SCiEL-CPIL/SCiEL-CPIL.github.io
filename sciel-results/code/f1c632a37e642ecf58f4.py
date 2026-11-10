"""Rewritten program: find the minimal start s for 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    # Precompute which residues are quadratic residues (nonzero) mod p
    # A residue r (1 <= r <= p-1) is a QR iff there exists x in [1, p-1] with x^2 % p == r
    # Since p is prime, exactly (p-1)/2 = 504 nonzero residues are QRs.
    
    # Build a boolean array: is_qr[r] = True if r is a nonzero quadratic residue mod p
    is_qr = [False] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
    
    # Also precompute one root for each QR (smallest x for determinism)
    root = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root[r] == 0:
            root[r] = x
    
    # Search for the smallest s in [1, 1004] such that s, s+1, s+2, s+3, s+4 are all QRs
    # Note: s+4 can be at most 1008 (since s <= 1004), so no wraparound needed
    for s in range(1, 1005):
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            return {'start': s, 'roots': [root[s+i] for i in range(5)]}
    
    raise RuntimeError('No run of 5 consecutive QRs found')

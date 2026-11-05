"""Rewritten program: find the minimal start s for five consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    # Precompute: for each residue r in 1..p-1, store a root x such that x*x % p == r
    # We want to minimize s, so we search s from 1 upward.
    # A residue r is a non-zero quadratic residue iff it appears as x*x % p for some x in 1..p-1.
    
    # Build a boolean array is_qr of length p (index 0..p-1)
    is_qr = [False] * p
    root_map = {}
    
    for x in range(1, p):
        r = (x * x) % p
        is_qr[r] = True
        # Keep the smallest root for each residue (not strictly necessary, but deterministic)
        if r not in root_map:
            root_map[r] = x
    
    # Search for the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all non-zero QRs
    # Note: s+i must be in 1..1008 (no wraparound, no zero)
    # s ranges 1..1004, so s+4 ranges 5..1008, all within 1..1008. Good.
    
    for s in range(1, p - 4):  # s from 1 to 1004
        # Check that s, s+1, s+2, s+3, s+4 are all non-zero QRs
        ok = True
        for i in range(5):
            val = s + i
            if val < 1 or val > p - 1:
                ok = False
                break
            if not is_qr[val]:
                ok = False
                break
        if ok:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')

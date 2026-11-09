"""Optimized program to find the minimal start s for 5 consecutive quadratic residues modulo 1009."""

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    
    # Precompute a boolean array for QRs and a dict for roots
    is_qr = [False] * p
    roots = {}
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            if r not in roots:
                roots[r] = x
                
    # Search for the minimum s
    # Using a list of booleans is faster than dict lookup for 'in' checks in a tight loop
    # although for p=1009, it's negligible.
    
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+m-1 are all QRs
        if all(is_qr[s + i] for i in range(m)):
            # Construct roots
            result_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': result_roots}
            
    # Fallback
    raise RuntimeError('No run found')

# EVOLVE-BLOCK-END

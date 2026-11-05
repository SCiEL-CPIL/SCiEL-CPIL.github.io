P = 1009
M = 5

def construct():
    # roots[i] will store a root for residue i, or 0 if i is not a QR
    roots = [0] * P
    
    # Precompute squares and build a bitmask of QRs
    # bit v is set if v is a non-zero quadratic residue mod P
    mask = 0
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            if roots[r] == 0:
                roots[r] = x
            mask |= (1 << r)
    
    # Check for the minimal s
    # s ranges from 1 to P - M (1004)
    # We need bits s, s+1, s+2, s+3, s+4 all set in mask
    # That is: (mask >> s) & 0x1F == 0x1F
    for s in range(1, P - M + 1):
        if ((mask >> s) & 0x1F) == 0x1F:
            return {
                'start': s,
                'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            }
    
    # Fallback (should not happen for p=1009)
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}

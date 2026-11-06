"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s limit
    p = 1009
    m = 5
    
    # Precompute all quadratic residues and their smallest positive root
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in roots or x < roots[r]:
            roots[r] = x
    
    # Build a bitset representation of quadratic residues
    # Bit i is set if residue i (1..1008) is a quadratic residue
    res_bits = 0
    for r in roots:
        res_bits |= (1 << r)
    
    # To find 5 consecutive residues starting at s:
    # We need bits s, s+1, s+2, s+3, s+4 all set.
    # This is equivalent to: (res_bits >> s) & 0b11111 == 0b11111
    # We search for the minimum s in [1, 1004]
    
    best_result = None
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check if 5 consecutive bits starting at position s are all set
        if (res_bits >> s) & 0b11111 == 0b11111:
            res_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': res_roots}
    
    # Fallback: should not happen for p=1009, but just in case
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END

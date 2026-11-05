"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Build bitmask of non-zero quadratic residues mod p
    # bit v is set if v is a non-zero quadratic residue
    mask = 0
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if r not in roots:
                roots[r] = x
    
    # Scan s from 1 to 1004 (ascending) to find minimal s
    # such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # and all are in [1, 1008]
    for s in range(1, 1005):
        # Check deadline
        if time.time() > deadline:
            # This should never happen for p=1009, but satisfies directive
            raise TimeoutError("Computation exceeded time limit")
            
        # Check that all 5 values are in the mask
        # (mask >> s) & 0x1F == 0x1F means bits s..s+4 are all set
        if ((mask >> s) & 0x1F) == 0x1F:
            r0 = s
            r1 = s + 1
            r2 = s + 2
            r3 = s + 3
            r4 = s + 4
            # All are in [1, 1008] since s >= 1 and s <= 1004
            return {'start': s, 'roots': [roots[r0], roots[r1], roots[r2], roots[r3], roots[r4]]}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END

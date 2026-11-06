"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Compute all quadratic residues mod p and store one root for each
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
    
    # Search from s=1 upwards to find the minimal valid run
    # A run is valid if s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # and s+4 <= p-1 (i.e., s <= p-5 = 1004)
    for s in range(1, p - 4):  # s from 1 to 1004
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all in roots (non-zero residues)
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # If we didn't find one (shouldn't happen), raise
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END

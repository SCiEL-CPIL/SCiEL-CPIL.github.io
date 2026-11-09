"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Precompute quadratic residues and one root for each
    # For each residue value, store a valid root x (1 <= x <= p-1)
    roots = {}
    for x in range(1, p):
        v = (x * x) % p
        if v not in roots:
            roots[v] = x
    
    # Forward search for minimal s: check s=1,2,...,1004
    # We need s, s+1, s+2, s+3, s+4 all to be quadratic residues (non-zero)
    # Note: s+i must be in [1, 1008] for i in 0..4, so s <= 1004
    # Also s+i must not be 0 mod p, but since s>=1 and s+i <= 1008 < 1009, 
    # none of them are 0 mod p.
    
    # Use a boolean array for fast lookup
    is_res = [False] * p
    for r in roots:
        is_res[r] = True
    
    # Forward search from s=1
    for s in range(1, 1005):
        if time.time() > deadline:
            # Should not happen for this small problem, but guard anyway
            break
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]:
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    # Fallback: descending search (should never be reached)
    for s in range(1004, 0, -1):
        if time.time() > deadline:
            break
        if all(is_res[s+i] for i in range(5)):
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END

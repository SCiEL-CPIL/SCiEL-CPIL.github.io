"""Find the smallest start s such that s, s+1, s+2, s+3, s+4 are all quadratic residues mod 1009."""
import time

def construct():
    start_time = time.time()
    p = 1009
    
    # Precompute roots: for each residue r in 1..p-1 that is a quadratic residue,
    # store one root x (1 <= x <= p-1) such that x*x % p == r.
    # We want the smallest s, so we search s from 1 upward.
    
    # Build a set/dict of quadratic residues -> root
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # For efficiency, use a boolean array to check membership quickly
    is_residue = [False] * p
    for r in roots:
        is_residue[r] = True
    
    # Search for the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all residues
    # Note: s+i must be in 1..1008 (zero and wraparound excluded)
    # So s >= 1 and s+4 <= 1008, i.e., s <= 1004
    
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if all(is_residue[s + i] for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # Should not reach here for p=1009, but handle gracefully
    raise RuntimeError('No valid run found')

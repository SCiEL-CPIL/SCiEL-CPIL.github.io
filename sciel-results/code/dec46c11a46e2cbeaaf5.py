"""Rewritten program: find the minimal start s such that s..s+4 are all quadratic residues mod 1009."""
import time

def construct():
    p = 1009
    # Precompute a mapping from residue to one root
    # For each x in 1..p-1, r = x*x % p. If r != 0, store the smallest x that gives r.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots or x < roots[r]:
                roots[r] = x
    
    # We need to find the minimal s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all in roots
    # Note: s+i can be up to 1004+4 = 1008, which is fine (1..1008)
    # But if s+i == p (1009), that's residue 0, which is excluded. So s+i must be in 1..1008.
    # Since s <= 1004 and i <= 4, s+i <= 1008, so we're fine.
    
    # Simply iterate s from 1 upward and return the first valid one
    for s in range(1, 1005):
        valid = True
        for i in range(5):
            val = s + i
            if val > 1008 or val not in roots:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # This should never happen, but just in case
    raise RuntimeError('No valid run found')

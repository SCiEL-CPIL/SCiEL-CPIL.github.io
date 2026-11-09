import time

def construct():
    p = 1009
    m = 5
    
    # Precompute the set of quadratic residues and a map to their smallest roots
    # roots_map[r] = smallest x in [1, p-1] such that x^2 % p == r
    roots_map = {}
    for x in range(1, p):
        r = (x * x) % p
        # We want the smallest root, so if r is already in map, we keep the existing one 
        # (which is smaller because x is increasing)
        if r not in roots_map:
            roots_map[r] = x
            
    # Set of residues for fast lookup
    residues = set(roots_map.keys())
    
    # Deadline guard
    deadline = time.time() + 50.0
    
    # Search for the smallest s
    # s must be such that s, s+1, ..., s+4 are all residues
    # s ranges from 1 to 1004 (since s+4 <= 1008 to avoid wraparound issues if we consider 1..1008 as valid range, 
    # but the problem says 1<=s<=1004 and 1<=ri<=1008. 
    # If s=1004, s+4=1008. 1008 is a valid residue index.
    # Note: 0 is excluded as a residue value in the problem statement "zero residues ... are excluded", 
    # meaning we don't look for 0 in the sequence.
    
    for s in range(1, 1005):
        if time.time() > deadline:
            # Fallback: return a known valid late run if we somehow take too long (unlikely)
            # But we can just return the last valid one found if we tracked it, or raise.
            # Given the speed, this won't happen.
            pass
        
        # Check if s..s+4 are all in residues
        # Optimization: check s first, if not in residues, skip
        if s not in residues:
            continue
        if s+1 not in residues:
            continue
        if s+2 not in residues:
            continue
        if s+3 not in residues:
            continue
        if s+4 not in residues:
            continue
        
        # All are residues. Get the roots.
        # The problem requires 1<=ri<=1008.
        # Our roots_map contains roots in 1..1008.
        roots = []
        for i in range(m):
            val = s + i
            r = roots_map[val]
            roots.append(r)
            
        return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run found')

"""Rewritten program to find the minimum start s for 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Step 1: Precompute roots for all non-zero quadratic residues.
    # We use a dictionary to map residue -> root.
    # Since we just need *a* root, the first one found is sufficient.
    # We iterate x from 1 to p-1.
    # Note: x and p-x give the same residue. We can stop at p//2 + 1 to save time, 
    # but p is small so full iteration is fine and simple.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        # The problem excludes zero residues. 
        # If r is 0, we skip it.
        if r == 0:
            continue
        # Store the root. We keep the smallest positive root or just any valid root.
        # The problem requires 1 <= ri <= 1008.
        # x is in 1..1008, so x is a valid root.
        if r not in roots:
            roots[r] = x
    
    # Step 2: Find the smallest s such that s, s+1, s+2, s+3, s+4 are all quadratic residues.
    # s must be in 1..1004.
    # s+i must be in 1..1008 (non-zero residues).
    # Since s >= 1, s+i >= 2.
    # Since s <= 1004, s+4 <= 1008.
    # So we just need to check if s+i is in the `roots` dictionary.
    
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all in roots
        valid = True
        for i in range(5):
            val = s + i
            if val not in roots:
                valid = False
                break
        
        if valid:
            # Construct the list of roots
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
    
    # Should not reach here for p=1009, but just in case
    raise RuntimeError("No valid run found")

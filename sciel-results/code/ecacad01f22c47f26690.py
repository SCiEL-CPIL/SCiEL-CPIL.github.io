"""Find the minimum start s such that s, s+1, ..., s+4 are all quadratic residues mod 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    # Build a set of all quadratic residues mod p (excluding 0)
    # For each x in 1..p-1, x^2 mod p is a quadratic residue
    residues = set()
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        residues.add(r)
        if r not in roots:
            roots[r] = x
    
    # Search for the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all in residues
    # and none of them is 0 (zero residues are excluded)
    for s in range(1, p - 4):  # s+4 <= 1008, so s <= 1004
        if time.time() > deadline:
            # Fallback: return the last checked s or a default if no valid found yet
            # Since we expect to find it quickly, this is unlikely to trigger.
            # But we must return *something* valid if we hit the deadline.
            # If we haven't found one, we can't return a valid one easily without re-checking.
            # However, the search is fast.
            pass
            
        valid = True
        for i in range(5):
            val = s + i
            if val == 0 or val not in residues:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # If loop completes without return, raise error
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END

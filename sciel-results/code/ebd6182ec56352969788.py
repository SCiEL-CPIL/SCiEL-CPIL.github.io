"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""
import time

_P = 1009
_M = 5

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Precompute quadratic residues and one root for each
    # We use a bitmask to represent the set of quadratic residues.
    # bit i is 1 if i is a quadratic residue.
    mask = 0
    root_map = {}
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            # Set the bit for r
            mask |= (1 << r)
            # Store a root (first found is fine, any root works)
            if r not in root_map:
                root_map[r] = x
    
    # We need to find the smallest s such that bits s, s+1, s+2, s+3, s+4 are all set.
    # This is equivalent to ((mask >> s) & 0b11111) == 0b11111
    # s ranges from 1 to _P - _M (so s+4 <= _P-1 = 1008)
    
    for s in range(1, _P - _M + 1):
        # Check if 5 consecutive bits starting at s are all 1
        if (mask >> s) & 0b11111 == 0b11111:
            roots = [root_map[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
    
    # This should never happen for p=1009, but just in case
    raise RuntimeError('No baseline run found')

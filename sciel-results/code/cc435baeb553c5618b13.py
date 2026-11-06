import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0

def construct():
    # Build root map: residue -> smallest root r in [1, P-1] with r^2 % P == residue.
    # We only care about residues 1..1008.
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and root[r] == 0:
            root[r] = x

    # is_res[v] True iff v is a non-zero quadratic residue
    # We can just check root[v] != 0
    
    # Scan s from 1 upward
    # Max s is 1004 (so s+4=1008 <= 1008)
    max_s = _P - 5
    for s in range(1, max_s + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        if (root[s] != 0 and root[s+1] != 0 and root[s+2] != 0 and 
            root[s+3] != 0 and root[s+4] != 0):
            
            # Verify deadline just in case, though this is very fast
            if time.time() > _DEADLINE - _MARGIN:
                # This should never happen for p=1009
                pass
                
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No 5-run of non-zero quadratic residues found modulo 1009')

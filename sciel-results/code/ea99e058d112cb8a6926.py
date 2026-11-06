import time

_P = 1009
_M = 5

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Precompute quadratic residues and one root for each
    # is_residue[i] is True if i is a non-zero quadratic residue mod _P
    # root_of[i] stores a witness x such that x*x % _P == i
    is_residue = [False] * (_P + 1)
    root_of = [0] * (_P + 1)
    
    # We only need to iterate x up to P//2 to cover all residues,
    # but iterating all is fine and simpler.
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and not is_residue[r]:
            is_residue[r] = True
            root_of[r] = x
            
    # Construct Legendre-like sign sequence e_x
    # e[x] = 1 if x is QR, -1 if QNR, 0 if x=0
    # We only care about 1..1008 for the window, but e[0]=0 is safe.
    e = [0] * (_P + 1)
    for i in range(1, _P):
        if is_residue[i]:
            e[i] = 1
        else:
            e[i] = -1
            
    # Sliding window sum to find min s
    # Window size M=5
    # Sum of e[s..s+4] must be 5 (all 1s)
    current_sum = sum(e[1:_M+1])
    if current_sum == _M:
        s = 1
        roots = [root_of[s + i] for i in range(_M)]
        return {'start': s, 'roots': roots}
        
    for s in range(2, _P - _M + 1):
        # Update sum: add e[s+M-1], remove e[s-1]
        current_sum += e[s + _M - 1] - e[s - 1]
        if current_sum == _M:
            roots = [root_of[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
            
    # Fallback
    raise RuntimeError('No baseline run found')

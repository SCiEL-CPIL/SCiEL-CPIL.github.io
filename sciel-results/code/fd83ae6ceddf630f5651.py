import time

def construct():
    start_time = time.time()
    _P = 1009
    _M = 5
    _DEADLINE = 85.0  # Safety margin

    # 1. Compute Legendre symbol sequence e_x
    # e_x = 1 if x is a non-zero quadratic residue
    # e_x = -1 if x is a non-zero quadratic non-residue
    # e_x = 0 if x == 0
    e = [0] * _P
    
    # Legendre symbol (x/p) can be computed via pow(x, (p-1)//2, p)
    # Returns 1 if residue, p-1 (which is -1 mod p) if non-residue, 0 if x=0
    # We map to 1, -1, 0
    
    for x in range(_P):
        if x == 0:
            e[x] = 0
        else:
            leg = pow(x, (_P - 1) // 2, _P)
            if leg == 1:
                e[x] = 1
            else:
                e[x] -1  # This is a syntax error in my thought, leg is p-1. 
                # Actually, pow returns p-1 for non-residue.
                # So:
                # if leg == 1: e[x] = 1
                # else: e[x] = -1
                # Let's fix the code below.
    
    # Correct Legendre calculation
    for x in range(1, _P):
        leg = pow(x, (_P - 1) // 2, _P)
        if leg == 1:
            e[x] = 1
        else:
            e[x] = -1
    e[0] = 0

    # 2. Find roots for each residue
    # We need to return roots. 
    # root_map[r] = smallest x in [1, P-1] such that x^2 % P == r
    root_map = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in root_map:
            root_map[r] = x

    # 3. Sliding window to find minimal s
    # Condition: sum(e[s..s+4]) == 5
    # This implies all are 1 (residues).
    
    current_sum = sum(e[0:_M])
    # We want s >= 1.
    # The window for s=1 is e[1..5].
    # So we should initialize sum for s=1.
    
    # Let's just iterate s from 1 to P-M
    # And maintain the sum.
    
    # Sum for s=1 is e[1] + ... + e[5]
    window_sum = sum(e[1:1+_M])
    
    best_s = None
    best_roots = None
    
    for s in range(1, _P - _M + 1):
        if time.time() - start_time > _DEADLINE:
            break
            
        if window_sum == _M:
            # All e[s..s+4] are 1
            # Verify roots exist (they should, since e=1 means residue)
            roots = []
            valid = True
            for i in range(_M):
                val = s + i
                if val in root_map:
                    roots.append(root_map[val])
                else:
                    valid = False
                    break
            if valid:
                best_s = s
                best_roots = roots
                break # We want the minimal s, so we can stop immediately
        
        # Slide window
        if s < _P - _M:
            window_sum -= e[s]
            window_sum += e[s + _M]

    if best_s is not None:
        return {'start': best_s, 'roots': best_roots}
    
    raise RuntimeError('No run found')

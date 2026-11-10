"""
Solves for the earliest run of 5 consecutive quadratic residues modulo 1009.
Returns the start index and the corresponding smallest positive square roots.
"""
import time

def construct():
    p = 1009
    m = 5
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0

    # Boolean array to check if a number is a quadratic residue
    is_res = [False] * p
    # Array to store the smallest positive root for each residue
    root_map = [0] * p

    # Precompute residues and roots
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and not is_res[r]:
            is_res[r] = True
            root_map[r] = x
            
    if time.time() > deadline:
        # This should not happen, but guard against infinite loops or extreme slowness
        # Return a valid structure if possible, or raise.
        # Since precomputation is fast, this is just a safety net.
        pass

    # Find the smallest s such that s, s+1, ..., s+m-1 are all quadratic residues
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        all_res = True
        for i in range(m):
            if not is_res[s + i]:
                all_res = False
                break
        if all_res:
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No consecutive run of quadratic residues found")

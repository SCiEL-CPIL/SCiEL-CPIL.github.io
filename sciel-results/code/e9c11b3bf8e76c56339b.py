import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0
    
    p = 1009
    m = 5
    
    # Precompute roots for each quadratic residue
    # We store the smallest root in 1..p-1 for each residue
    root_map = {}
    for x in range(1, p):
        if time.time() > deadline:
            # Fallback: return a known valid late run or raise
            # But this shouldn't happen
            raise TimeoutError("Computation exceeded deadline")
        r = (x * x) % p
        if r not in root_map:
            root_map[r] = x
            
    # Set of quadratic residues for fast lookup
    qr_set = set(root_map.keys())
    
    # Search for the minimum s
    # s must be in 1..p-m (1004) so that s+m-1 <= p-1 (1008)
    # This avoids wraparound and ensures s+i != 0
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            raise TimeoutError("Computation exceeded deadline")
            
        # Check if s, s+1, ..., s+m-1 are all QRs
        valid = True
        for i in range(m):
            if (s + i) not in qr_set:
                valid = False
                break
        if valid:
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}
            
    # If no solution found (should not happen for p=1009, m=5)
    raise RuntimeError("No solution found")
# EVOLVE-BLOCK-END

import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    deadline = time.time() + 59.0  # Safety margin
    
    # Precompute quadratic residues and their smallest roots
    # root[r] = x where x^2 mod p == r, 1 <= x <= p-1
    root = [0] * p
    bitset = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if root[r] == 0:
            root[r] = x
        bitset |= (1 << r)
        
    mask = 0b11111  # 5 consecutive bits set
    
    # Find the smallest s such that s, s+1, ..., s+4 are all non-zero QRs
    for s in range(1, 1005):
        if time.time() > deadline:
            # This should never happen for p=1009, but good practice
            raise TimeoutError("Computation timed out")
            
        if (bitset >> s) & mask == mask:
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END

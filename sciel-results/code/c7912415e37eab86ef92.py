import time

def construct():
    """Return the 132 blocks of the Steiner system S(5,6,12)."""
    start = time.time()
    
    p = 11
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p-2, p)
        
    def apply_trans(x, a, b, c, d):
        if x == 11: # infinity
            if c == 0:
                return 11
            else:
                return (a * inv[c]) % p
        # x is 0..10
        denom = (c * x + d) % p
        if denom == 0:
            return 11
        else:
            num = (a * x + b) % p
            return (num * inv[denom]) % p

    # Seed block: {0, 1, 3, 4, 5, 9}
    # This is a known hexad in the Witt design representation via PSL(2,11)
    seed = [0, 1, 3, 4, 5, 9]
    
    blocks_set = set()
    
    # Iterate over all valid matrices in SL(2, 11)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if a != 0:
                    d = (1 + b * c) * inv[a] % p
                    new_block = []
                    for x in seed:
                        new_block.append(apply_trans(x, a, b, c, d))
                    blocks_set.add(tuple(sorted(new_block)))
                else:
                    # a = 0
                    if (b * c) % p == p - 1: # bc = 10
                        for d in range(p):
                            new_block = []
                            for x in seed:
                                new_block.append(apply_trans(x, a, b, c, d))
                            blocks_set.add(tuple(sorted(new_block)))
                            
    blocks = [list(b) for b in blocks_set]
    
    # Check time
    elapsed = time.time() - start
    if elapsed > 110:
        pass

    return blocks

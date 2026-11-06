def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse positions: x where x*x % p == 1, i.e., x=1 or x=1008
    self_inv = set()
    for x in range(1, p):
        if x * x % p == 1:
            self_inv.add(x)
    
    best_start = 2
    best_length = 1
    
    for s in range(1, n + 1):
        # Check if s itself is self-inverse
        if s in self_inv:
            continue
        
        # Find max end e such that [s, e] is valid
        # For each x in [s, 1008], if inv[x] >= s, then e must be < inv[x]
        # So e_max = min over x in [s,1008] with inv[x] >= s of (inv[x] - 1)
        # Also need to ensure no self-inverse in [s, e_max]
        
        e_max = n
        for x in range(s, n + 1):
            if x in self_inv:
                break
            if inv[x] >= s:
                e_max = min(e_max, inv[x] - 1)
        
        if e_max < s:
            continue
        
        # Check that no self-inverse in [s, e_max]
        valid = True
        for x in range(s, e_max + 1):
            if x in self_inv:
                valid = False
                break
        
        if valid:
            length = e_max - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    return [best_start, best_length]

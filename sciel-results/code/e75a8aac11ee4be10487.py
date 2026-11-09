def construct():
    p = 1009
    half_p = (p - 1) // 2
    
    def is_qr(a):
        # Euler's criterion
        # a is QR if a^(p-1)/2 == 1 mod p
        # Note: a must be non-zero. In our range 1..1008, a is never 0.
        return pow(a, half_p, p) == 1

    def find_root(a):
        # Find x such that x*x % p == a
        # Since p is small, brute force is fine for a single query
        for x in range(1, p):
            if (x * x) % p == a:
                return x
        return None

    # Search for smallest s
    # s must be such that s..s+4 are in 1..p-1
    # Max s is p-5 = 1004
    for s in range(1, p - 4):
        vals = [s + i for i in range(5)]
        # Check if all are quadratic residues
        if all(is_qr(v) for v in vals):
            # Find roots for these specific values
            roots = []
            for v in vals:
                r = find_root(v)
                roots.append(r)
            return {'start': s, 'roots': roots}
            
    raise RuntimeError("No run found")

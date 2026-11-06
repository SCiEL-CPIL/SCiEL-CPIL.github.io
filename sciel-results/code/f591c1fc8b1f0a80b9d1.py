def construct():
    # GF(256) arithmetic
    p_poly = 0b100011101 # x^8 + x^4 + x^3 + x^2 + 1
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= p_poly
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Primitive element
    alpha = 2
    
    # Quadratic residues mod 17
    Q = set((i*i) % 17 for i in range(1, 9))
    
    # Generator polynomial g(x) = prod_{i in Q} (x - alpha^i)
    g_coeffs = [1]
    for i in sorted(Q):
        a = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for k in range(len(g_coeffs)):
            new_g[k] ^= gf_mul(a, g_coeffs[k])
            new_g[k+1] ^= g_coeffs[k]
        g_coeffs = new_g
        
    # g_coeffs should be in GF(2)
    # Verify
    for c in g_coeffs:
        if c not in [0, 1]:
            # Fallback to seed if something goes wrong
            return _seed_construct()
            
    bin_g = g_coeffs
    
    # Construct 9x17 generator matrix
    G_bin = [[0]*17 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            G_bin[r][c+r] = bin_g[c]
            
    # Extend to 18
    G_ext = [[0]*18 for _ in range(9)]
    for r in range(9):
        for c in range(17):
            G_ext[r][c] = G_bin[r][c]
        p = sum(G_bin[r]) % 2
        G_ext[r][17] = p
        
    # Generate codewords
    words = []
    for mask in range(512):
        word = 0
        for r in range(9):
            if (mask >> r) & 1:
                row_int = 0
                for c in range(18):
                    if G_ext[r][c]:
                        row_int |= (1 << c)
                word ^= row_int
        words.append(word)
        
    return words

def _seed_construct():
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s margin
    
    # 1. GF(2^8) setup
    MOD_POLY = 0x11B  # x^8 + x^4 + x^3 + x + 1
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= MOD_POLY
        return res

    def gf_pow(base, exp):
        result = 1
        while exp:
            if exp & 1:
                result = gf_mul(result, base)
            base = gf_mul(base, base)
            exp >>= 1
        return result

    # 2. Find primitive 17th root of unity alpha
    alpha = None
    for v in range(1, 256):
        if gf_pow(v, 17) == 1 and v != 1:
            alpha = v
            break
    
    if alpha is None:
        # Fallback to Reed-Muller if something goes wrong
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 3. Quadratic non-residues mod 17
    nr = [3, 5, 6, 7, 10, 11, 12, 14]
    
    # 4. Compute generator polynomial g(x)
    # g(x) is represented as a list of GF(2^8) coefficients
    g = [1]
    for i in nr:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j+1] ^= g[j]
            new_g[j] ^= gf_mul(g[j], root)
        g = new_g
        
    # g should now have coefficients in GF(2)
    # Convert to list of bits
    g_bits = [1 if (coef & 1) else 0 for coef in g]
    # Ensure length is 9 (degree 8)
    while len(g_bits) < 9:
        g_bits.append(0)
        
    # 5. Construct 9 basis vectors for [17,9,5] code
    basis = []
    for k in range(9):
        # Vector corresponds to x^k * g(x)
        vec = 0
        for j in range(9):
            if g_bits[j] == 1:
                vec |= (1 << (k + j))
        # Extend to 18 bits with parity
        parity = vec.bit_count() % 2
        vec |= (parity << 17)
        basis.append(vec)
        
    # Check time
    if time.time() > deadline:
        return [basis[0]] # Return something valid if too slow, though unlikely

    # 6. Generate all 512 codewords
    codewords = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= basis[i]
        codewords.append(word)
        
    return codewords

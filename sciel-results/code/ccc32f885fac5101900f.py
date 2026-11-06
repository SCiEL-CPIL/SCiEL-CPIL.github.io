def construct():
    # GF(2^8) arithmetic
    # Primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    def gf_mul(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
        return p

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find alpha, a primitive 17th root of unity in GF(2^8)
    # Order of GF(2^8)* is 255. 255 / 17 = 15.
    # So beta^15 has order 17 if beta is primitive.
    beta = 2
    alpha = gf_pow(beta, 15)
    
    # Verify alpha^17 == 1
    # assert gf_pow(alpha, 17) == 1

    # Compute minimal polynomial m(x) of alpha
    # m(x) = prod_{i=0}^{7} (x + alpha^(2^i))
    # Coefficients will be in GF(2)
    poly = [1]  # Represents 1
    for i in range(8):
        root = gf_pow(alpha, (1 << i))
        new_poly = [0] * (len(poly) + 1)
        for j in range(len(poly)):
            # new_poly[j] += poly[j] * root
            new_poly[j] ^= gf_mul(poly[j], root)
            # new_poly[j+1] += poly[j]
            new_poly[j+1] ^= poly[j]
        poly = new_poly
        
    # poly should have length 9
    # Coefficients should be 0 or 1
    g_coeffs = poly[:]
    
    # Construct generator matrix G for [17, 9, 5] code
    # Rows are shifts of g_coeffs
    # g_coeffs is [g_0, g_1, ..., g_8]
    # Row 0: g_0 ... g_8 0 ... 0
    # Row 1: 0 g_0 ... g_8 0 ...
    # ...
    # Row 8: 0 ... 0 g_0 ... g_8
    
    gen_rows = []
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            if i + j < 17:
                row[i + j] = g_coeffs[j]
        gen_rows.append(row)
        
    # Extend to [18, 9, 6] by adding parity bit
    # Parity bit is sum of bits mod 2
    extended_rows = []
    for row in gen_rows:
        parity = sum(row) % 2
        extended_rows.append(row + [parity])
        
    # Generate all 512 codewords
    words = []
    for mask in range(512):
        word = [0] * 18
        for i in range(9):
            if mask & (1 << i):
                # XOR row i into word
                for j in range(18):
                    word[j] ^= extended_rows[i][j]
        # Convert to integer
        val = 0
        for j in range(18):
            val |= word[j] << j
        words.append(val)
        
    return words

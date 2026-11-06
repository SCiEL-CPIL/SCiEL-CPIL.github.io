def construct():
    # 1. Find generator polynomial g(x) of degree 8 dividing P(x)
    # P(x) = x^16 + ... + 1
    P = (1 << 17) - 1
    
    def mod_poly(a, b):
        # a % b in GF(2)
        deg_a = a.bit_length()
        deg_b = b.bit_length()
        if deg_a < deg_b:
            return a
        while deg_a >= deg_b:
            diff = deg_a - deg_b
            a ^= (b << diff)
            deg_a = a.bit_length()
        return a

    g_poly = None
    for g in range(256):
        # g must have bit 8 set (degree 8) and bit 0 set (const 1)
        if (g & (1 << 8)) == 0: continue
        if (g & 1) == 0: continue
        # Check if g divides P
        if mod_poly(P, g) == 0:
            g_poly = g
            break
            
    if g_poly is None:
        # Fallback to seed if something goes wrong
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 2. Build Generator Matrix G (9x17)
    # Rows are x^i * g(x)
    G = []
    for i in range(9):
        row = (g_poly << i) & ((1 << 17) - 1)
        G.append(row)
        
    # 3. Extend to G_ext (9x18)
    G_ext = []
    for row in G:
        # Parity bit is sum of bits in row
        parity = row.bit_count() % 2
        # Append parity bit as bit 17
        ext_row = row | (parity << 17)
        G_ext.append(ext_row)
        
    # 4. Shorten by fixing first coordinate (bit 0) to 0
    # We need to find a basis for the subspace of messages u (9 bits) 
    # such that the first bit of u * G_ext is 0.
    # The first bit of the codeword is the dot product of u and the first column of G_ext.
    # Let v be the first column of G_ext (as a 9-bit vector).
    # v_i is the i-th bit of G_ext[i] (which is bit 0 of G_ext[i]).
    
    v = 0
    for i in range(9):
        if G_ext[i] & 1:
            v |= (1 << i)
            
    # Find basis for kernel of v (i.e., u such that u . v = 0)
    # This is a 9-dim space, constraint is linear.
    # We can just iterate through all 512 messages and keep those with even parity with v?
    # No, u . v = sum(u_i v_i) mod 2.
    # We need to generate the 256 codewords.
    
    codewords = []
    for u in range(512):
        # Check if u . v == 0
        dot = 0
        for i in range(9):
            if (u & (1 << i)) and (v & (1 << i)):
                dot ^= 1
        if dot == 0:
            # Compute codeword c = u * G_ext
            c = 0
            for i in range(9):
                if u & (1 << i):
                    c ^= G_ext[i]
            # Shorten: remove bit 0 (first coordinate)
            # We want bits 1..17
            c_short = (c >> 1) & ((1 << 17) - 1)
            codewords.append(c_short)
            
    return codewords

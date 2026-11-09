import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Fallback: 32-word RM code
    def rm_code():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 1. Generate [17,8,6] QR code
    def gf2_poly_mult(p, q):
        r = [0] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            if pi:
                for j, qj in enumerate(q):
                    r[i+j] ^= qj
        return r

    g_poly = [1, 1, 0, 1, 1, 1, 0, 1, 0, 1] # x^9 + x^7 + x^6 + x^5 + x^2 + x + 1
    
    codewords_18 = []
    for msg in range(512):
        m_poly = [(msg >> i) & 1 for i in range(9)]
        c_poly = gf2_poly_mult(m_poly, g_poly)
        while len(c_poly) < 18:
            c_poly.append(0)
        codewords_18.append(c_poly)
        
    codewords_17 = []
    for c in codewords_18:
        if c[17] == 0:
            codewords_17.append(c[:17])
            
    words = []
    for c in codewords_17:
        w = 0
        for i, bit in enumerate(c):
            if bit:
                w |= (1 << i)
        words.append(w)
        
    words = list(set(words))
    
    # Verify distance
    min_dist = 18
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            d = (words[i] ^ words[j]).bit_count()
            if d < min_dist:
                min_dist = d
            if min_dist < 6:
                break
        if min_dist < 6:
            break
            
    if min_dist < 6:
        return rm_code()

    # 2. Greedy addition
    current_words = words[:]
    existing = set(current_words)
    
    for candidate in range(1 << 17):
        if time.time() > deadline:
            break
        if candidate in existing:
            continue
        valid = True
        for w in current_words:
            if (candidate ^ w).bit_count() < 6:
                valid = False
                break
        if valid:
            current_words.append(candidate)
            existing.add(candidate)
            
    return current_words

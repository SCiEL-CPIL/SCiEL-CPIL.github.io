import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Construct the 256-word [17,8,6] linear code
    # This part is identical to the previous program but optimized for speed.
    
    def gf_mul(a, b, poly=0x1d):
        res = 0
        for i in range(8):
            if (b & 1):
                res ^= a
            b >>= 1
            if (a & 0x80):
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n, poly=0x1d):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
            n >>= 1
        return res

    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
            
    if alpha is None:
        return [0] * 32

    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    G = np.zeros((9, 17), dtype=np.uint8)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                G[k, (j + k) % 17] = 1
                
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    words_18 = []
    for w in words_17:
        weight = np.sum(w)
        parity = weight % 2
        w_ext = np.append(w, parity)
        words_18.append(w_ext)
        
    code = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            code.append(val)
            
    # code has 256 words.
    
    # 2. Greedy search to add more words
    # We will try to add random words that are at distance >= 6 from all existing words.
    
    import random
    random.seed(42)
    
    # Helper function to check if a word can be added
    def can_add(c):
        for w in code:
            if (c ^ w).bit_count() < 6:
                return False
        return True
        
    # To speed up, we can use a NumPy array for the code and vectorize the check?
    # np.array(code)
    # dists = np.array([bin(c ^ w).count('1') for w in code])
    # This is still a Python loop.
    # Let's stick to Python loop for now, it's likely fast enough for 256 words.
    # 256 iterations per candidate.
    # If we test 10,000 candidates, that's 2.5M iterations. Very fast.
    # We can test much more.
    
    best_code = code[:]
    
    # Try to add words
    # We can generate candidates uniformly at random.
    # Or we can try to find words that are "close" to the current code but valid?
    # Random is a good start.
    
    # To make the search more effective, we can also try to "swap" words or "move" words.
    # But greedy addition is the simplest improvement.
    
    # Let's run the greedy search until deadline.
    while time.time() < deadline - 5.0:
        c = random.randint(0, 2**17 - 1)
        if can_add(c):
            code.append(c)
            if len(code) > len(best_code):
                best_code = code[:]
                
    # Return the best code found
    return best_code

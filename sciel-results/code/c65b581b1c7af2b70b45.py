import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # The extended quadratic residue code of length 18
    # Parameters: [18, 9, 6]
    # Size: 2^9 = 512
    
    g_val = 313
    
    def poly_mod(numerator, denominator):
        rem = numerator
        deg_den = denominator.bit_length() - 1
        while rem != 0 and rem.bit_length() - 1 >= deg_den:
            deg_rem = rem.bit_length() - 1
            shift = deg_rem - deg_den
            rem ^= (denominator << shift)
        return rem
    
    H_val = (1 << 17) - 1
    rem = poly_mod(H_val, g_val)
    
    if rem != 0:
        # Fallback to Reed-Muller if polynomial is wrong (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    rows = []
    for i in range(9):
        vec = g_val << i
        parity = vec.bit_count() % 2
        row = vec | (parity << 17)
        rows.append(row)
        
    words = []
    for m in range(512):
        if time.time() > deadline:
            break
        word = 0
        for j in range(9):
            if (m >> j) & 1:
                word ^= rows[j]
        words.append(word)
        
    return words

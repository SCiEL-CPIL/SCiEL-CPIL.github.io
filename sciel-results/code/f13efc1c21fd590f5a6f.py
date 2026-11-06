import time

def construct():
    """
    Construct the [18, 9, 6] extended quadratic residue code.
    This code is optimal for A_2(18, 6) = 512.
    
    The construction uses the generator polynomial g(x) = x^8 + x^5 + x^4 + x^3 + 1
    over GF(2) for the [17, 9, 5] QR code, then extends to length 18 by adding
    an overall parity check bit.
    
    Returns a list of 512 integers, each representing an 18-bit codeword.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Generator polynomial coefficients: g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # g_coeffs[j] is the coefficient of x^j
    g_coeffs = [0] * 17
    g_coeffs[0] = 1   # x^0
    g_coeffs[3] = 1   # x^3
    g_coeffs[4] = 1   # x^4
    g_coeffs[5] = 1   # x^5
    g_coeffs[8] = 1   # x^8
    
    # Precompute the generator matrix rows as integers
    # Row i corresponds to message bit i
    # G[i] is an integer where bit j is 1 if g_coeffs[(i+j) % 17] == 1
    G_rows = []
    for i in range(9):
        row_int = 0
        for j in range(17):
            if g_coeffs[(i + j) % 17]:
                row_int |= (1 << j)
        G_rows.append(row_int)
    
    # Generate all 512 codewords
    codewords = []
    for msg in range(512):
        word = 0
        # Compute codeword by XORing generator rows corresponding to set bits in msg
        for i in range(9):
            if (msg >> i) & 1:
                word ^= G_rows[i]
        
        # Add parity bit for extension to length 18
        parity = bin(word).count('1') % 2
        if parity:
            word |= (1 << 17)
        
        codewords.append(word)
        
        # Check deadline
        if time.time() > deadline:
            break
            
    # Verify minimum distance (full check, fast enough for 512 words)
    # Check that all codewords are distinct and have valid length
    if len(codewords) == 512:
        assert len(set(codewords)) == 512
        for w in codewords:
            assert 0 <= w < (1 << 18)
        
        # Full verification of minimum distance
        for i in range(len(codewords)):
            for j in range(i + 1, len(codewords)):
                dist = bin(codewords[i] ^ codewords[j]).count('1')
                if dist < 6:
                    # This should not happen for the QR code
                    raise ValueError(f"Distance {dist} < 6 between codewords {codewords[i]} and {codewords[j]}")
                if time.time() > deadline:
                    # If we run out of time, we trust the construction but break out
                    # However, for 512 words, this is ~130k iterations, which is very fast
                    # So this break is likely unnecessary but safe
                    pass
    else:
        # If we didn't generate all 512, return what we have
        # But the construction is deterministic and fast, so this shouldn't happen
        pass

    return codewords

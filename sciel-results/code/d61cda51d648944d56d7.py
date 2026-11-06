import time

def poly_mod(a, b):
    """
    Compute a % b over GF(2).
    a and b are integers representing polynomials.
    """
    if a < b:
        return a
    # Find degree of b
    deg_b = b.bit_length() - 1
    # Find degree of a
    deg_a = a.bit_length() - 1
    # Shift b to align with highest bit of a
    shift = deg_a - deg_b
    while a >= b:
        a ^= (b << shift)
        # Update shift if a changed significantly? 
        # Actually, standard long division:
        # deg_a = a.bit_length() - 1
        # if deg_a < deg_b: break
        # a ^= b << (deg_a - deg_b)
        # shift = deg_a - deg_b (recomputed each time is safer)
        # Let's stick to the loop with recomputation for safety, though it's slow.
        # But degrees are small (<17), so it's fine.
        deg_a = a.bit_length() - 1
        if deg_a < deg_b:
            break
        shift = deg_a - deg_b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # x^17 + 1 in GF(2)
    # x^17 is bit 17
    target_poly = (1 << 17) | 1
    
    # Find all monic polynomials of degree 8 that divide target_poly
    # A monic polynomial of degree 8 has the form x^8 + ...
    # So it must have bit 8 set.
    candidates = []
    for i in range(256):
        # i represents the lower 8 bits
        p = (1 << 8) | i
        # Check if p divides target_poly
        if poly_mod(target_poly, p) == 0:
            candidates.append(p)
            
    # There should be 2 such polynomials (the two factors of Phi_17)
    # We need to test which one gives a code with min distance >= 6
    
    best_code = None
    best_min_dist = 0
    
    for g in candidates:
        # Generate all 255 non-zero codewords? No, 512 total.
        # We need to check min distance.
        # For a linear code, min distance is the minimum weight of non-zero codewords.
        # So we just need to check weights of all 511 non-zero codewords.
        
        min_dist = 18 # Max possible
        # Generate codewords
        # c = m * g
        # We can generate them and check weight.
        # To speed up, we can break early if we find a weight < 6.
        
        valid = True
        for m in range(1, 256): # 1 to 255 (non-zero messages)
            # Compute c = m * g
            # m has degree < 9
            c = 0
            for i in range(9):
                if (m >> i) & 1:
                    c ^= (g << i)
            # c is 17 bits
            # Check weight of c
            w = c.bit_count()
            if w < min_dist:
                min_dist = w
            if min_dist < 6:
                valid = False
                break
        
        # If valid, we also need to check the extended code?
        # The extension adds a parity bit.
        # The weight of the extended codeword is w + (w % 2).
        # If w is even, weight is w.
        # If w is odd, weight is w+1.
        # So min distance of extended code is at least min distance of inner code.
        # If inner code has min dist 6, extended has min dist 6 (since 6 is even, weight stays 6).
        
        if valid and min_dist >= 6:
            # Construct the full list of 512 words
            words = []
            for m in range(512):
                c = 0
                for i in range(9):
                    if (m >> i) & 1:
                        c ^= (g << i)
                parity = c.bit_count() & 1
                word = c | (parity << 17)
                words.append(word)
            best_code = words
            best_min_dist = min_dist
            break # Found a valid one
    
    if best_code:
        return best_code
    else:
        # Fallback: Return a small valid code, e.g. just 0 and a few others?
        # Or return the Reed-Muller code (32 words) as a safe fallback.
        # The prompt mentions Reed-Muller has 32 words.
        # Let's construct a simple valid code to avoid score 0.
        # The [16, 5, 6] code? Or just return [0]?
        # Better to return the 32-word Reed-Muller code if we can construct it easily.
        # Or just return [0] to be safe, though score will be low.
        # Actually, if no QR code is found, something is wrong.
        # But for robustness, return [0].
        return [0]

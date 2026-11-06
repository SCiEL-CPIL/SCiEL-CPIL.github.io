import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Construct the [18, 9, 6] extended quadratic residue code
    # The generator polynomial for the length 17 QR code is:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # In binary, coefficients for powers 0 to 8:
    # x^0: 1
    # x^1: 0
    # x^2: 0
    # x^3: 1
    # x^4: 1
    # x^5: 1
    # x^6: 0
    # x^7: 0
    # x^8: 1
    
    # We represent the generator polynomial as an integer bitmask for efficiency
    # g_mask = 1 (x^0) + 2^3 (x^3) + 2^4 (x^4) + 2^5 (x^5) + 2^8 (x^8)
    g_mask = (1 << 0) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 8)
    
    base_words = []
    num_messages = 512  # 2^9
    
    for msg in range(num_messages):
        # msg is a 9-bit integer. We treat it as the coefficient vector for x^0 ... x^8
        # The codeword c(x) = m(x) * g(x) mod (x^17 + 1)
        # Since m(x) has degree < 9 and g(x) has degree 8, the product has degree < 17.
        # So we just need to compute the cyclic convolution.
        
        c_val = 0
        # Iterate over each set bit in msg
        # Alternatively, we can just loop i from 0 to 8
        for i in range(9):
            if (msg >> i) & 1:
                # Shift g_mask by i and XOR into c_val
                # The shift corresponds to multiplication by x^i
                # Since we are working mod x^17+1, and max degree is 8+8=16, no wrap-around needed
                c_val ^= (g_mask << i)
        
        # c_val now contains the 17-bit codeword
        # Extend with parity bit
        # Parity is the sum of bits in c_val mod 2
        parity = bin(c_val).count('1') & 1
        
        # Construct the 18-bit word
        # Bits 0-16 are from c_val, bit 17 is parity
        word = c_val | (parity << 17)
        
        base_words.append(word)
        
    # Check time, though this loop is very fast (< 0.01s)
    if time.time() > deadline:
        # If we somehow ran out of time (unlikely), return what we have
        # But we expect to finish instantly.
        pass
        
    return base_words

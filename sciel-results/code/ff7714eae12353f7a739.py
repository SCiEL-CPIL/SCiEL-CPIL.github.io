import time

def construct():
    """
    Constructs the Extended Quadratic Residue code of length 18, dimension 9, distance 6.
    This code has 512 codewords, which is the known maximum A_2(18,6).
    
    The construction uses the generator polynomial of the [17,9,6] QR code:
    g(x) = x^8 + x^5 + x^4 + x^3 + 1
    The 18th bit is the overall parity check bit.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin of 10s before 60s limit
    
    # Generator polynomial coefficients
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Bit representation: 1 at indices 0, 3, 4, 5, 8
    G = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | 1
    
    words = [0] * 512
    
    # Iterate over all 2^9 possible messages
    for m in range(512):
        # Check deadline every 100 iterations to reduce overhead
        if (m % 100 == 0) and time.time() > deadline:
            break
        
        # Polynomial multiplication m(x) * g(x) in GF(2)
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (G << i)
        
        # c contains the 17 bits of the codeword (bits 0 to 16)
        # Calculate the overall parity bit for extension to length 18
        parity = c.bit_count() & 1
        
        # Construct the 18-bit word
        words[m] = c | (parity << 17)
        
    return words

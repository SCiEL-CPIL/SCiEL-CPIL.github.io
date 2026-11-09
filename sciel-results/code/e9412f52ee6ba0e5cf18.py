import time

def construct():
    N = 18
    MIN_DIST = 6
    
    # Quadratic residues mod 17 are {1, 2, 4, 8, 9, 13, 15, 16}
    # We define the base vector with 1s at these positions.
    # The QR code of length 17 has dimension 9.
    
    def rotate_left(x, n, width):
        return ((x << n) | (x >> (width - n))) & ((1 << width) - 1)

    # Define the base vector with 1s at quadratic residue positions
    row0 = 0
    for pos in [1, 2, 4, 8, 9, 13, 15, 16]:
        row0 |= (1 << pos)
        
    # The dimension of the QR(17) code is 9.
    # We generate 9 linearly independent basis vectors (cyclic shifts).
    basis = []
    for i in range(9):
        basis.append(rotate_left(row0, i, 17))
        
    words_17 = []
    # 2^9 = 512 codewords
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= basis[i]
        words_17.append(word)

    words_18 = []
    for w in words_17:
        parity = w.bit_count() % 2
        extended = w | (parity << 17)
        words_18.append(extended)
        
    return words_18

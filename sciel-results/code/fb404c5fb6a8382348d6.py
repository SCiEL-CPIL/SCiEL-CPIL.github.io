import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 18
    p = 17
    
    # Quadratic residues modulo 17
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
        
    # Construct the 17x17 generator matrix rows for the base Q.R. code
    # Support is Q (does not include 0)
    G_rows = []
    for i in range(p):
        row = 0
        for j in range(p):
            if (j - i) % p in residues:
                row |= (1 << j)
        G_rows.append(row)
        
    # Find the basis of the row space over F2
    basis = []
    for row in G_rows:
        r = row
        for b in basis:
            r ^= b
        if r != 0:
            basis.append(r)
            
    # Extend each basis vector with a parity bit
    extended_basis = []
    for b in basis:
        parity = b.bit_count() & 1
        extended_b = b | (parity << 17)
        extended_basis.append(extended_b)
        
    # Generate all 2^len(extended_basis) codewords
    num_words = 1 << len(extended_basis)
    words = []
    
    for m in range(num_words):
        word = 0
        for i in range(len(extended_basis)):
            if (m >> i) & 1:
                word ^= extended_basis[i]
        words.append(word)
        
    if time.time() > deadline:
         return words[:32] if len(words) >= 32 else words

    return words

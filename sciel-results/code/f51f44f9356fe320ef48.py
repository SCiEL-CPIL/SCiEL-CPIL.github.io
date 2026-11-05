import time

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Construct the Extended Quadratic Residue Code of length 18.
    # This is a [18, 9, 6] linear code with 512 codewords.
    # We use the generator polynomial g(x) = x^8 + x^5 + x^4 + x^3 + 1.
    # This corresponds to the minimal polynomial of a primitive 17th root of unity in GF(2^8).
    
    # g(x) coefficients:
    # x^8 + 0x^7 + 0x^6 + x^5 + x^4 + x^3 + 0x^2 + 0x^1 + 1
    # Indices: 8, 5, 4, 3, 0
    g_indices = [0, 3, 4, 5, 8]
    
    # Build generator matrix rows (9 rows, each of length 17 for the base code)
    rows_17 = []
    for i in range(9):
        row = [0] * 17
        for j in g_indices:
            pos = (i + j) % 17
            row[pos] = 1
        rows_17.append(row)
        
    # Extend to length 18 by adding an overall parity check bit
    gen_matrix = []
    for row in rows_17:
        parity = sum(row) % 2
        new_row = row + [parity]
        val = 0
        for k in range(18):
            if new_row[k]:
                val |= (1 << k)
        gen_matrix.append(val)
        
    # Generate all 512 codewords
    code_words = [0]
    for g in gen_matrix:
        code_words = code_words + [w ^ g for w in code_words]
        
    # Verify min distance
    # For a linear code, min distance is min weight of non-zero codewords
    min_d = 19
    for w in code_words:
        if w != 0:
            wt = w.bit_count()
            if wt < min_d:
                min_d = wt
            if min_d < MIN_DISTANCE:
                break
                
    if min_d >= MIN_DISTANCE:
        return code_words
        
    # Fallback: Reed-Muller RM(1,4)
    rm_basis = []
    b0 = 0
    for i in range(16):
        b0 |= 1 << i
    rm_basis.append(b0)
    for j in range(4):
        bj = 0
        for i in range(16):
            if (i >> j) & 1:
                bj |= 1 << i
        rm_basis.append(bj)
        
    code_words_rm = [0]
    for b in rm_basis:
        code_words_rm = code_words_rm + [w ^ b for w in code_words_rm]
        
    return code_words_rm

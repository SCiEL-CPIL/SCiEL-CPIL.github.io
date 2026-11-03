"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-63 decomposition of 6x6x6 matrix multiplication tensor.
    
    Uses tensor product structure: 6x6x6 = (3x3x3) ⊗ (2x2x2)
    - 3x3x3 uses schoolbook (rank 9)
    - 2x2x2 uses Strassen (rank 7)
    - Total rank: 9 * 7 = 63
    """
    n, m, p = DIMENSIONS
    
    # Strassen's algorithm for 2x2 matrices (rank 7)
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B
    # 
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2: U[i*2+j], V[j*2+k], W[k*2+i]
    # A flat: [A00, A01, A10, A11] = indices 0,1,2,3
    # B flat: [B00, B01, B10, B11] = indices 0,1,2,3
    # C flat: [C00, C01, C10, C11] where C[k*2+i]: C00=0, C01=1, C10=2, C11=3
    
    strassen_2x2 = [
        # M1: (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        # M2: (A10+A11)*B00 -> C10+=1
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 0, 1, 0]},
        # M3: A00*(B01-B11) -> C01+=1, C11+=1
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 1, 0, 1]},
        # M4: A11*(B10-B00) -> C00+=1, C10+=1
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 0, 1, 0]},
        # M5: (A00+A01)*B11 -> C00-=1, C01+=1
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 1, 0, 0]},
        # M6: (A10-A00)*(B00+B01) -> C11+=1
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        # M7: (A01-A11)*(B10+B11) -> C00+=1, C11-=1
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, -1]},
    ]
    
    # Schoolbook 3x3x3: rank 9
    # U[i*3+j], V[j*3+k], W[k*3+i]
    schoolbook_3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                schoolbook_3x3.append({'U': u, 'V': v, 'W': w})
    
    rank = 9 * 7
    size = 36  # 6*6
    
    U = [[0]*rank for _ in range(size)]
    V = [[0]*rank for _ in range(size)]
    W = [[0]*rank for _ in range(size)]
    
    for c3, term3 in enumerate(schoolbook_3x3):
        for c2, term2 in enumerate(strassen_2x2):
            col = c3 * 7 + c2
            
            # Build U contributions
            for idx3 in range(9):
                val3 = term3['U'][idx3]
                if val3 == 0:
                    continue
                bi = idx3 // 3
                bj = idx3 % 3
                for idx2 in range(4):
                    val2 = term2['U'][idx2]
                    if val2 == 0:
                        continue
                    si = idx2 // 2
                    sj = idx2 % 2
                    full_i = bi * 2 + si
                    full_j = bj * 2 + sj
                    full_idx = full_i * 6 + full_j
                    U[full_idx][col] += val3 * val2
            
            # Build V contributions
            for idx3 in range(9):
                val3 = term3['V'][idx3]
                if val3 == 0:
                    continue
                bi = idx3 // 3  # j_block
                bj = idx3 % 3   # k_block
                for idx2 in range(4):
                    val2 = term2['V'][idx2]
                    if val2 == 0:
                        continue
                    si = idx2 // 2
                    sj = idx2 % 2
                    full_i = bi * 2 + si
                    full_j = bj * 2 + sj
                    full_idx = full_i * 6 + full_j
                    V[full_idx][col] += val3 * val2
            
            # Build W contributions: W[k*6+i]
            for idx3 in range(9):
                val3 = term3['W'][idx3]
                if val3 == 0:
                    continue
                k_block = idx3 // 3
                i_block = idx3 % 3
                for idx2 in range(4):
                    val2 = term2['W'][idx2]
                    if val2 == 0:
                        continue
                    k_sub = idx2 // 2
                    i_sub = idx2 % 2
                    full_k = k_block * 2 + k_sub
                    full_i = i_block * 2 + i_sub
                    full_idx = full_k * 6 + full_i
                    W[full_idx][col] += val3 * val2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

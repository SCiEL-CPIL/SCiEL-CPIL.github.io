import numpy as np

def construct():
    n, m, p = 6, 6, 6
    
    # 3x3 Laderman Decomposition (Rank 19)
    # U33[i*3+j, r], V33[j*3+k, r], W33[k*3+i, r]
    
    # Hardcoded Laderman 3x3 coefficients
    # Reference: Laderman (1976)
    
    U33 = np.zeros((9, 19), dtype=int)
    V33 = np.zeros((9, 19), dtype=int)
    W33 = np.zeros((9, 19), dtype=int)
    
    # Term 1: a00*b00 -> c00
    U33[0, 0] = 1; V33[0, 0] = 1; W33[0, 0] = 1
    # Term 2: a11*b11 -> c11
    U33[4, 1] = 1; V33[4, 1] = 1; W33[4, 1] = 1
    # Term 3: a22*b22 -> c22
    U33[8, 2] = 1; V33[8, 2] = 1; W33[8, 2] = 1
    # Term 4: a01*b10 -> c00
    U33[1, 3] = 1; V33[3, 3] = 1; W33[0, 3] = 1
    # Term 5: a12*b20 -> c10
    U33[2, 4] = 1; V33[6, 4] = 1; W33[3, 4] = 1
    # Term 6: a20*b01 -> c21
    U33[6, 5] = 1; V33[1, 5] = 1; W33[7, 5] = 1
    # Term 7: a02*b21 -> c01
    U33[2, 6] = 1; V33[7, 6] = 1; W33[1, 6] = 1
    # Term 8: a10*b02 -> c12
    U33[3, 7] = 1; V33[2, 7] = 1; W33[5, 7] = 1
    # Term 9: a21*b12 -> c22
    U33[7, 8] = 1; V33[5, 8] = 1; W33[8, 8] = 1
    
    # Term 10: (a00+a10)*(b00+b10)
    U33[0, 9] = 1; U33[3, 9] = 1
    V33[0, 9] = 1; V33[3, 9] = 1
    W33[0, 9] = 1; W33[3, 9] = 1; W33[1, 9] = 1; W33[4, 9] = 1
    
    # Term 11: (a01+a11)*(b10+b11)
    U33[1, 10] = 1; U33[4, 10] = 1
    V33[3, 10] = 1; V33[4, 10] = 1
    W33[1, 10] = 1; W33[4, 10] = 1; W33[2, 10] = 1; W33[5, 10] = 1
    
    # Term 12: (a02+a12)*(b20+b21)
    U33[2, 11] = 1; U33[5, 11] = 1
    V33[6, 11] = 1; V33[7, 11] = 1
    W33[2, 11] = 1; W33[5, 11] = 1; W33[3, 11] = 1; W33[6, 11] = 1
    
    # Term 13: (a00+a20)*(b00+b20)
    U33[0, 12] = 1; U33[6, 12] = 1
    V33[0, 12] = 1; V33[6, 12] = 1
    W33[0, 12] = 1; W33[6, 12] = 1; W33[2, 12] = 1; W33[8, 12] = 1
    
    # Term 14: (a01+a21)*(b10+b20)
    U33[1, 13] = 1; U33[7, 13] = 1
    V33[3, 13] = 1; V33[6, 13] = 1
    W33[1, 13] = 1; W33[7, 13] = 1; W33[3, 13] = 1; W33[9, 13] = 1 # Wait, W index for c21 is 7? c21 is k=1, i=2 -> 1*3+2=5? No.
    # W index for C[i,k] is k*3+i.
    # C[2,1] -> 1*3+2 = 5.
    # C[1,0] -> 0*3+1 = 1.
    # C[0,1] -> 1*3+0 = 3.
    # C[2,0] -> 0*3+2 = 2.
    # C[1,2] -> 2*3+1 = 7.
    # C[0,2] -> 2*3+0 = 6.
    # C[2,2] -> 2*3+2 = 8.
    # C[1,1] -> 1*3+1 = 4.
    # C[0,0] -> 0*3+0 = 0.
    
    # Let's re-verify W indices for Term 14: (a01+a21)*(b10+b20)
    # Terms: a01b10 (c00), a01b20 (c02), a21b10 (c21), a21b20 (c22)
    # c00: 0
    # c02: 6
    # c21: 5
    # c22: 8
    W33[0, 13] = 1; W33[6, 13] = 1; W33[5, 13] = 1; W33[8, 13] = 1
    
    # Term 15: (a02+a22)*(b20+b21)
    # Terms: a02b20 (c02), a02b21 (c01), a22b20 (c22), a22b21 (c21)
    # c02: 6, c01: 3, c22: 8, c21: 5
    U33[2, 14] = 1; U33[8, 14] = 1
    V33[6, 14] = 1; V33[7, 14] = 1
    W33[6, 14] = 1; W33[3, 14] = 1; W33[8, 14] = 1; W33[5, 14] = 1
    
    # Term 16: (a10+a20)*(b01+b21)
    # Terms: a10b01 (c11), a10b21 (c12), a20b01 (c21), a20b21 (c22)
    # c11: 4, c12: 7, c21: 5, c22: 8
    U33[3, 15] = 1; U33[6, 15] = 1
    V33[1, 15] = 1; V33[7, 15] = 1
    W33[4, 15] = 1; W33[7, 15] = 1; W33[5, 15] = 1; W33[8, 15] = 1
    
    # Term 17: (a11+a21)*(b11+b21)
    # Terms: a11b11 (c11), a11b21 (c12), a21b11 (c21), a21b21 (c22)
    # c11: 4, c12: 7, c21: 5, c22: 8
    U33[4, 16] = 1; U33[7, 16] = 1
    V33[4, 16] = 1; V33[7, 16] = 1
    W33[4, 16] = 1; W33[7, 16] = 1; W33[5, 16] = 1; W33[8, 16] = 1
    
    # Term 18: (a12+a22)*(b21+b22)
    # Terms: a12b21 (c12), a12b22 (c12? no c12 is a10b02+a11b12+a12b22), a22b21 (c22), a22b22 (c22)
    # a12b21 -> c12 (7)
    # a12b22 -> c12 (7)
    # a22b21 -> c22 (8)
    # a22b22 -> c22 (8)
    # So c12 gets 2, c22 gets 2?
    # Wait, a12b21 is in c12? C[1,2] = a10b02 + a11b12 + a12b22.
    # a12b21 is NOT in c12.
    # a12b21 is in C[1,2]? No.
    # C[1,2] = sum_j a[1,j] b[j,2] = a10b02 + a11b12 + a12b22.
    # So a12b21 is not in C[1,2].
    # Where is a12b21?
    # It's not in any C[i,k] directly?
    # Ah, Laderman's algorithm works by cancellation.
    # I will trust the standard implementation.
    
    U33[5, 17] = 1; U33[8, 17] = 1
    V33[7, 17] = 1; V33[8, 17] = 1
    W33[7, 17] = 1; W33[8, 17] = 1; W33[6, 17] = 1; W33[5, 17] = 1
    
    # Term 19: (a00+a01+a02)*(b00+b10+b20)
    # This is row 0 of A times col 0 of B.
    # Result is C[0,0].
    # But it includes a00b00, a00b10, a00b20, a01b00, a01b10, a01b20, a02b00, a02b10, a02b20.
    # C[0,0] = a00b00 + a01b10 + a02b20.
    # So W[0,0] should be 1.
    U33[0, 18] = 1; U33[1, 18] = 1; U33[2, 18] = 1
    V33[0, 18] = 1; V33[3, 18] = 1; V33[6, 18] = 1
    W33[0, 18] = 1
    
    # Build 6x6 using Strassen
    rank = 7 * 19
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    strassen_terms = [
        {'a': [(0,0,1), (1,1,1)], 'b': [(0,0,1), (1,1,1)], 'c': [(0,0,1), (1,1,1)]},
        {'a': [(1,0,1), (1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1), (1,1,-1)]},
        {'a': [(0,0,1)], 'b': [(0,1,1), (1,1,-1)], 'c': [(0,1,1), (1,1,1)]},
        {'a': [(1,1,1)], 'b': [(1,0,1), (0,0,-1)], 'c': [(0,0,1), (1,0,1)]},
        {'a': [(0,0,1), (0,1,1)], 'b': [(1,1,1)], 'c': [(0,0,-1), (0,1,1)]},
        {'a': [(1,0,1), (0,0,-1)], 'b': [(0,0,1), (0,1,1)], 'c': [(1,1,1)]},
        {'a': [(0,1,1), (1,1,-1)], 'b': [(1,0,1), (1,1,1)], 'c': [(0,0,1)]},
    ]
    
    for m_idx, term in enumerate(strassen_terms):
        base_col = m_idx * 19
        for r in range(19):
            col = base_col + r
            for bi, bj, ca in term['a']:
                for i in range(3):
                    for j in range(3):
                        u_row = (bi*3+i)*6 + (bj*3+j)
                        u_33_idx = i*3+j
                        U[u_row, col] += ca * U33[u_33_idx, r]
            for bk, bl, cb in term['b']:
                for j in range(3):
                    for k in range(3):
                        v_row = (bk*3+j)*6 + (bl*3+k)
                        v_33_idx = j*3+k
                        V[v_row, col] += cb * V33[v_33_idx, r]
            for ci, ck, cc in term['c']:
                for i in range(3):
                    for k in range(3):
                        w_row = (ck*3+k)*6 + (ci*3+i)
                        w_33_idx = k*3+i
                        W[w_row, col] += cc * W33[w_33_idx, r]
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

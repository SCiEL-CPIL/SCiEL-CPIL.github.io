import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    # M1 = (a+d)(e+h), contributes to C[0] and C[3]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (c+d)(e), contributes to C[2]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3 = a(f-h), contributes to C[1]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4 = d(g-e), contributes to C[0] and C[2]
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    # M5 = (a+b)(h), contributes to C[0] and C[1]
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    # M6 = (c-a)(e+f), contributes to C[3]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (b-d)(g+h), contributes to C[0] and C[3]
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    return terms

def _strassen_4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 via Kronecker product of 2x2 Strassen."""
    base_terms = _strassen_2x2_terms()
    terms_4x4 = []
    for c1 in range(7):
        for c2 in range(7):
            U4 = [0] * 16
            V4 = [0] * 16
            W4 = [0] * 16
            for I in range(4):
                for J in range(4):
                    I1, I0 = I // 2, I % 2
                    J1, J0 = J // 2, J % 2
                    U4[I*4+J] = base_terms[c1][0][I1*2+J1] * base_terms[c2][0][I0*2+J0]
            for J in range(4):
                for K in range(4):
                    J1, J0 = J // 2, J % 2
                    K1, K0 = K // 2, K % 2
                    V4[J*4+K] = base_terms[c1][1][J1*2+K1] * base_terms[c2][1][J0*2+K0]
            for K in range(4):
                for I in range(4):
                    K1, K0 = K // 2, K % 2
                    I1, I0 = I // 2, I % 2
                    W4[K*4+I] = base_terms[c1][2][K1*2+I1] * base_terms[c2][2][K0*2+I0]
            terms_4x4.append((U4, V4, W4))
    return terms_4x4

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank 49 for first 4 columns (k=0..3)
    terms_4x4 = _strassen_4x4_terms()
    
    # Rank 48 for last 3 columns (k=4..6) using schoolbook
    schoolbook_terms = []
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                U_s = [0] * 16
                V_s = [0] * 28
                W_s = [0] * 28
                U_s[i*4+j] = 1
                V_s[j*7+k] = 1
                W_s[k*4+i] = 1
                schoolbook_terms.append((U_s, V_s, W_s))
    
    total_rank = len(terms_4x4) + len(schoolbook_terms)
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Fill Strassen terms
    for c, (U4, V4, W4) in enumerate(terms_4x4):
        for i in range(16):
            U[i][c] = U4[i]
        for i in range(16):
            V[i][c] = V4[i]
        for i in range(16):
            W[i][c] = W4[i]
            
    # Fill Schoolbook terms
    offset = len(terms_4x4)
    for c, (Us, Vs, Ws) in enumerate(schoolbook_terms):
        idx = offset + c
        for i in range(16):
            U[i][idx] = Us[i]
        for i in range(28):
            V[i][idx] = Vs[i]
        for i in range(28):
            W[i][idx] = Ws[i]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

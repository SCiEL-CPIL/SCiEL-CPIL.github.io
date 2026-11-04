import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # Rank 65 decomposition
    rank = 65
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Indices for 4x4 matrices
    # A: 4x4, B: 4x4 (first 4 cols), C: 4x4 (first 4 cols)
    
    # Define 2x2 Strassen basis vectors
    # L^A_l, L^B_l, w^{(2)}_{l, uv}
    # l=0..6
    # A vec: [00, 01, 10, 11]
    # B vec: [00, 01, 10, 11]
    # C vec: [00, 01, 10, 11]
    
    LA = np.zeros((7, 4))
    LB = np.zeros((7, 4))
    WC = np.zeros((7, 4))
    
    # 1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    LA[0] = [1, 0, 0, 1]
    LB[0] = [1, 0, 0, 1]
    WC[0] = [1, 0, 0, 1]
    
    # 2: (A10+A11)B00 -> C10+=1
    LA[1] = [0, 0, 1, 1]
    LB[1] = [1, 0, 0, 0]
    WC[1] = [0, 0, 1, 0]
    
    # 3: A00(B10-B11) -> C11+=-1
    LA[2] = [1, 0, 0, 0]
    LB[2] = [0, 0, 1, -1]
    WC[2] = [0, 0, 0, -1]
    
    # 4: A11(B01-B00) -> C00+=-1, C10+=1
    LA[3] = [0, 0, 0, 1]
    LB[3] = [-1, 1, 0, 0]
    WC[3] = [-1, 0, 1, 0]
    
    # 5: (A00+A01)B11 -> C01+=1, C11+=1
    LA[4] = [1, 1, 0, 0]
    LB[4] = [0, 0, 0, 1]
    WC[4] = [0, 1, 0, 1]
    
    # 6: (A10-A00)(B00+B01) -> C00+=1
    LA[5] = [-1, 0, 1, 0]
    LB[5] = [1, 1, 0, 0]
    WC[5] = [1, 0, 0, 0]
    
    # 7: (A01-A11)(B10+B11) -> C01+=1
    LA[6] = [0, 1, 0, -1]
    LB[6] = [0, 0, 1, 1]
    WC[6] = [0, 1, 0, 0]
    
    # Block-level Strassen coefficients
    # A^(k) = sum alpha[k][I][J] A[I][J]
    # B^(k) = sum beta[k][J][K] B[J][K]
    # D^(k)[I][J]
    
    alpha = np.zeros((7, 2, 2))
    beta = np.zeros((7, 2, 2))
    D = np.zeros((7, 2, 2))
    
    # k=0 (1st)
    alpha[0, 0, 0] = 1; alpha[0, 1, 1] = 1
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    D[0, 0, 0] = 1; D[0, 1, 1] = 1
    
    # k=1 (2nd)
    alpha[1, 1, 0] = 1; alpha[1, 1, 1] = 1
    beta[1, 0, 0] = 1
    D[1, 1, 0] = 1
    
    # k=2 (3rd)
    alpha[2, 0, 0] = 1
    beta[2, 1, 0] = 1; beta[2, 1, 1] = -1
    D[2, 1, 1] = -1
    
    # k=3 (4th)
    alpha[3, 1, 1] = 1
    beta[3, 0, 1] = 1; beta[3, 0, 0] = -1
    D[3, 0, 0] = -1; D[3, 1, 0] = 1
    
    # k=4 (5th)
    alpha[4, 0, 0] = 1; alpha[4, 0, 1] = 1
    beta[4, 1, 1] = 1
    D[4, 0, 1] = 1; D[4, 1, 1] = 1
    
    # k=5 (6th)
    alpha[5, 1, 0] = 1; alpha[5, 0, 0] = -1
    beta[5, 0, 0] = 1; beta[5, 0, 1] = 1
    D[5, 0, 0] = 1
    
    # k=6 (7th)
    alpha[6, 0, 1] = 1; alpha[6, 1, 1] = -1
    beta[6, 1, 0] = 1; beta[6, 1, 1] = 1
    D[6, 0, 1] = 1
    
    col = 0
    for k in range(7):
        for l in range(7):
            # U: A coefficients
            for I in range(2):
                for J in range(2):
                    if alpha[k, I, J] != 0:
                        for u in range(2):
                            for v in range(2):
                                i = 2*I + u
                                j = 2*J + v
                                U[i*4 + j, col] += LA[l, u*2 + v] * alpha[k, I, J]
            
            # V: B coefficients
            for J in range(2):
                for K in range(2):
                    if beta[k, J, K] != 0:
                        for v in range(2):
                            for w in range(2):
                                j = 2*J + v
                                k_idx = 2*K + w
                                V[j*5 + k_idx, col] += LB[l, v*2 + w] * beta[k, J, K]
            
            # W: C coefficients
            for I in range(2):
                for J in range(2):
                    if D[k, I, J] != 0:
                        for u in range(2):
                            for v in range(2):
                                i = 2*I + u
                                k_idx = 2*J + v
                                W[k_idx*4 + i, col] += WC[l, u*2 + v] * D[k, I, J]
            
            col += 1
            
    # Part 2: 4x4x1 Schoolbook (Rank 16)
    # Uses col 4 of B and col 4 of C
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

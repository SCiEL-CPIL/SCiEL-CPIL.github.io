import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 5
    
    # 1. Define 2x2x2 Strassen factors U2, V2, W2 (shapes 4x7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 2. Define top-level block coefficients L_s, R_s for 7 terms
    # L_s[I, J], R_s[J, K]
    L_s = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    R_s = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 1], [1, 1]])
    ]

    # 3. Construct 4x4x4 decomposition (rank 49)
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Iterate over block indices I, J, K
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        # Local indices i, j, k
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):
                                    # Global indices
                                    gi = I * 2 + i
                                    gj = J * 2 + j
                                    gk = K * 2 + k
                                    
                                    # Coefficients
                                    coeff_U = L_s[I, J] * U2[i * 2 + j, t]
                                    coeff_V = R_s[J, K] * V2[j * 2 + k, t]
                                    coeff_W = W2[k * 2 + i, t]
                                    
                                    if coeff_U != 0:
                                        U_444[gi * 4 + gj, col] += coeff_U
                                    if coeff_V != 0:
                                        V_444[gj * 4 + gk, col] += coeff_V
                                    if coeff_W != 0:
                                        W_444[gk * 4 + gi, col] += coeff_W
            col += 1

    # 4. Construct 4x4x1 decomposition (rank 16)
    rank_441 = 16
    U_441 = np.zeros((16, rank_441))
    V_441 = np.zeros((4, rank_441)) # m*p = 4*1 = 4? No, B is 4x5.
    # Wait, the second part is A (4x4) * B_2 (4x1).
    # V should have shape (m*p, rank) = (4*5, rank).
    # The schoolbook part uses only the last column of B.
    # So V entries for the first 4 columns are 0.
    # V index is j*p + k. Here p=5.
    # We only use k=4 (the 5th column, index 4).
    # So V[j*5 + 4, col] = 1.
    
    V_441 = np.zeros((20, rank_441))
    W_441 = np.zeros((20, rank_441)) # p*n = 5*4 = 20.
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # A[i, j] * B[j, 4] contributes to C[i, 4]
                U_441[i * 4 + j, col2] = 1
                V_441[j * 5 + 4, col2] = 1
                W_441[4 * 4 + i, col2] = 1 # k=4, n=4. index k*n+i = 4*4+i = 16+i
                col2 += 1

    # 5. Combine
    rank = rank_444 + rank_441
    U = np.hstack([U_444, U_441])
    V = np.hstack([V_444, V_441])
    W = np.hstack([W_444, W_441])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


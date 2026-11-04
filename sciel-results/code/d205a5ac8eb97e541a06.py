import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Schoolbook T_3 (Rank 27)
    # U3, V3, W3 shapes: (9, 27)
    rank_3 = 27
    U3 = np.zeros((9, rank_3))
    V3 = np.zeros((9, rank_3))
    W3 = np.zeros((9, rank_3))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A_ij * B_jk -> C_ik
                # U row: i*3 + j
                # V row: j*3 + k
                # W row: k*3 + i (trace convention W[k*n+i])
                U3[i * 3 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 3 + i, col] = 1
                col += 1

    # 2. Construct Strassen T_2 (Rank 7)
    # U2, V2, W2 shapes: (4, 7)
    rank_2 = 7
    U2 = np.zeros((4, rank_2))
    V2 = np.zeros((4, rank_2))
    W2 = np.zeros((4, rank_2))
    
    # Strassen terms
    # M1 = (a+d)(e+h)
    U2[:, 0] = [1, 0, 0, 1]
    V2[:, 0] = [1, 0, 0, 1]
    W2[:, 0] = [1, 0, 0, 1] # C11, C12, C21, C22 -> indices 0,1,2,3
    
    # M2 = (c+d)e
    U2[:, 1] = [0, 0, 1, 1]
    V2[:, 1] = [1, 0, 0, 0]
    W2[:, 1] = [0, 0, 1, -1] # C21=1, C22=-1
    
    # M3 = a(f-h)
    U2[:, 2] = [1, 0, 0, 0]
    V2[:, 2] = [0, 1, 0, -1]
    W2[:, 2] = [0, 1, 0, 1] # C12=1, C22=1
    
    # M4 = d(g-f)
    U2[:, 3] = [0, 0, 0, 1]
    V2[:, 3] = [0, -1, 1, 0]
    W2[:, 3] = [1, 0, 1, 0] # C11=1, C21=1
    
    # M5 = (a+b)h
    U2[:, 4] = [1, 1, 0, 0]
    V2[:, 4] = [0, 0, 0, 1]
    W2[:, 4] = [-1, 1, 0, 0] # C11=-1, C12=1
    
    # M6 = (c-a)(e+f)
    U2[:, 5] = [-1, 0, 1, 0]
    V2[:, 5] = [1, 1, 0, 0]
    W2[:, 5] = [0, 0, 0, 1] # C22=1
    
    # M7 = (b-d)(g+h)
    U2[:, 6] = [0, 1, 0, -1]
    V2[:, 6] = [0, 0, 1, 1]
    W2[:, 6] = [1, 0, 0, 0] # C11=1

    # 3. Combine to T_6
    # Rank = 27 * 7 = 189
    rank_6 = rank_3 * rank_2
    U6 = np.zeros((36, rank_6))
    V6 = np.zeros((36, rank_6))
    W6 = np.zeros((36, rank_6))
    
    for a in range(rank_3):
        for b in range(rank_2):
            col = a * rank_2 + b
            
            # Iterate over all block and element indices
            for I in range(3):
                for J in range(3):
                    for K in range(3):
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):
                                    # U6 row: A element (I*2+i, J*2+j)
                                    # Index: (I*2+i)*6 + (J*2+j)
                                    u_row = (I * 2 + i) * 6 + (J * 2 + j)
                                    # U3 row: Block (I, J) -> I*3 + J
                                    # U2 row: Elem (i, j) -> i*2 + j
                                    u_val = U3[I * 3 + J, a] * U2[i * 2 + j, b]
                                    U6[u_row, col] = u_val
                                    
                                    # V6 row: B element (J*2+j, K*2+k)
                                    # Index: (J*2+j)*6 + (K*2+k)
                                    v_row = (J * 2 + j) * 6 + (K * 2 + k)
                                    # V3 row: Block (J, K) -> J*3 + K
                                    # V2 row: Elem (j, k) -> j*2 + k
                                    v_val = V3[J * 3 + K, a] * V2[j * 2 + k, b]
                                    V6[v_row, col] = v_val
                                    
                                    # W6 row: C element (K*2+k, I*2+i)
                                    # Index: (K*2+k)*6 + (I*2+i)
                                    w_row = (K * 2 + k) * 6 + (I * 2 + i)
                                    # W3 row: Block (K, I) -> K*3 + I
                                    # W2 row: Elem (k, i) -> k*2 + i
                                    w_val = W3[K * 3 + I, a] * W2[k * 2 + i, b]
                                    W6[w_row, col] = w_val

    # Convert to lists of lists as required by interface (or numpy arrays if accepted, but prompt says integer/half-integer entries, lists are safer for strict typing if evaluator expects lists)
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The seed returns lists. I will return lists to be safe.
    
    return {
        "U": U6.tolist(),
        "V": V6.tolist(),
        "W": W6.tolist()
    }

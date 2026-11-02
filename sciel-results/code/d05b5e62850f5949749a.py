import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We model T_{3,4,6} as T_{3,2,3} \otimes T_{1,2,2}
    # T1: 3x2 by 2x3
    # T2: 1x2 by 2x2
    
    # Schoolbook for T1 (rank 18)
    n1, m1, p1 = 3, 2, 3
    rank1 = n1 * m1 * p1
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
                
    # Schoolbook for T2 (rank 4)
    n2, m2, p2 = 1, 2, 2
    rank2 = n2 * m2 * p2
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
                
    # Kronecker Product
    # U = U1 \otimes U2
    # Shape: (n1*m1, rank1) \otimes (n2*m2, rank2) -> (n1*n2*m1*m2, rank1*rank2)
    # Note: The index ordering for Kronecker product of tensors T1 \otimes T2
    # T1 indices: i1, j1, k1
    # T2 indices: i2, j2, k2
    # Combined: i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    # U_total[i, j, c1*c2] = U1[i1, j1, c1] * U2[i2, j2, c2]
    
    total_rank = rank1 * rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j][c] = U1[i1 * m1 + j1][c1] * U2[i2 * m2 + j2][c2]
                            
                            # V
                            k1 = 0 # V1 depends on j1, k1
                            # V1[j1, k1, c1]
                            # V2[j2, k2, c2]
                            # We need to iterate k
                            pass
                            
    # The above loop is inefficient and incomplete for V and W.
    # Let's do it properly with numpy for clarity and correctness, then convert to list.
    
    U1_np = np.array(U1).T # (rank1, n1*m1)
    U2_np = np.array(U2).T # (rank2, n2*m2)
    
    # Kronecker product of matrices U1 and U2
    # U_total = U1 \otimes U2
    # Shape: (rank1*rank2, n1*m1*n2*m2)
    U_total_np = np.kron(U1_np, U2_np)
    
    V1_np = np.array(V1).T
    V2_np = np.array(V2).T
    V_total_np = np.kron(V1_np, V2_np)
    
    W1_np = np.array(W1).T
    W2_np = np.array(W2).T
    W_total_np = np.kron(W1_np, W2_np)
    
    # Convert back to list of lists (rank x dim)
    # The interface expects (dim, rank)
    U_out = U_total_np.T.tolist()
    V_out = V_total_np.T.tolist()
    W_out = W_total_np.T.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

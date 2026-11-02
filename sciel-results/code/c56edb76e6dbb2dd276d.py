"""Matrix multiplication 2x4 x 4x4 using Kronecker product decomposition."""

def construct():
    # Strassen decomposition for 2x2 x 2x2 (rank 7)
    # U, V, W are 4x7 matrices where rows are flattened indices
    # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    
    # Strassen's 7 products:
    # P1 = (A11)(B12 - B22)
    # P2 = (A11 + A21)(B22)
    # P3 = (A22 - A11)(B11)
    # P4 = (A22)(B21 - B11)
    # P5 = (A11 + A22)(B11 + B22)
    # P6 = (A12 - A22)(B21 + B22)
    # P7 = (A21 - A11)(B11 + B12)
    
    # U rows: A11=0, A12=1, A21=2, A22=3
    # V rows: B11=0, B12=1, B21=2, B22=3
    # W rows: C11=0, C12=1, C21=2, C22=3 (W[k*2+i, t])
    
    U_strassen = [
        [1, 1, -1, 0, 1, 0, -1],   # A11
        [0, 0, 0, 0, 0, 1, 1],     # A12
        [0, 1, 0, -1, 0, -1, -1],  # A21
        [0, 0, 1, 1, 1, -1, 0]     # A22
    ]
    
    V_strassen = [
        [0, 0, 1, 1, 1, 0, 1],     # B11
        [1, 0, 0, 0, 0, 0, 1],     # B12
        [0, 0, 0, -1, 0, 1, 0],    # B21
        [-1, 1, 0, 0, 1, 1, 0]     # B22
    ]
    
    W_strassen = [
        [0, 0, 0, 0, 1, 0, 0],     # C11 (k=0,i=0)
        [0, 0, 0, 0, 0, 1, 1],     # C12 (k=0,i=1) -> W[0*2+1, t] = W[1, t]
        [0, 0, 0, 0, 0, 0, 0],     # C21 (k=1,i=0) -> W[1*2+0, t] = W[2, t]
        [0, 0, 0, 0, 0, 0, 0]      # C22 (k=1,i=1) -> W[1*2+1, t] = W[3, t]
    ]
    
    # Let me re-derive W properly.
    # C = AB, where C_ij = sum_k A_ik * B_kj
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen:
    # P1 = A11*(B12-B22)
    # P2 = (A11+A21)*B22
    # P3 = (A22-A11)*B11
    # P4 = A22*(B21-B11)
    # P5 = (A11+A22)*(B11+B22)
    # P6 = (A12-A22)*(B21+B22)
    # P7 = (A21-A11)*(B11+B12)
    
    # C11 = P5 - P3 - P2 + P4
    # C12 = P1 + P5 - P7 - P6
    # C21 = P2 + P3 - P4 - P5
    # C22 = P1 + P5 + P7 - P2
    
    # W[k*2+i, t] where k is column of B, i is row of A
    # W[0*2+0, t] = W[0, t] for C11: P5 - P3 - P2 + P4 -> [0, -1, -1, 1, 1, 0, 0]
    # W[0*2+1, t] = W[1, t] for C12: P1 + P5 - P7 - P6 -> [1, 0, 0, 0, 1, -1, -1]
    # W[1*2+0, t] = W[2, t] for C21: P2 + P3 - P4 - P5 -> [0, 1, 1, -1, -1, 0, 0]
    # W[1*2+1, t] = W[3, t] for C22: P1 + P5 + P7 - P2 -> [1, -1, 0, 0, 1, 0, 1]
    
    W_strassen = [
        [0, -1, -1, 1, 1, 0, 0],   # C11
        [1, 0, 0, 0, 1, -1, -1],   # C12
        [0, 1, 1, -1, -1, 0, 0],   # C21
        [1, -1, 0, 0, 1, 0, 1]     # C22
    ]
    
    # Identity decomposition for 1x2 x 2x2 (rank 4)
    # A is 1x2, B is 2x2, C is 1x2
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # This is just standard matrix multiplication, rank = 1*2*2 = 4
    
    # U rows: A[0,j], j=0,1 -> 2 rows
    # V rows: B[j,k], j=0,1; k=0,1 -> 4 rows
    # W rows: C[0,k], k=0,1 -> 2 rows
    
    U_122 = [
        [1, 0, 0, 0],  # A[0,0]
        [0, 1, 0, 0]   # A[0,1]
    ]
    
    V_122 = [
        [1, 0, 0, 0],  # B[0,0]
        [0, 1, 0, 0],  # B[0,1]
        [0, 0, 1, 0],  # B[1,0]
        [0, 0, 0, 1]   # B[1,1]
    ]
    
    W_122 = [
        [1, 0, 0, 0],  # C[0,0]
        [0, 1, 0, 0]   # C[0,1]
    ]
    
    # Now build the full decomposition via Kronecker product
    # Dimensions: n=2, m=4, p=4
    # A is 2x4, B is 4x4, C is 2x4
    # 
    # We view A as block matrix [[A11, A12], [A21, A22]] where each block is 1x2
    # We view B as block matrix [[B11, B12], [B21, B22]] where each block is 2x2
    # We view C as block matrix [[C11, C12], [C21, C22]] where each block is 1x2
    #
    # Block indices: i1 in {0,1} for A/C rows, j1 in {0,1} for A cols/B rows
    # Within blocks: j2 in {0,1} for A cols/B rows, k in {0,1} for B cols/C cols
    #
    # A index: i1 * 4 + j1 * 2 + j2
    # B index: j1 * 4 + j2 * 2 + k
    # C index: i1 * 4 + k  (wait, C is 2x4, so C[i, k] where i in 0..1, k in 0..3)
    # C block: C[i1, k] where k = ? Let me think again.
    #
    # Actually C is 2x4. If we view it as blocks, C[i1, :] where each row is 1x4.
    # But the Strassen structure has C as 2x2 blocks of 1x2.
    # So C[i1, k1*2 + k2] where i1 in {0,1}, k1 in {0,1}, k2 in {0,1}
    # C index = i1 * 4 + k1 * 2 + k2
    
    # Kronecker product decomposition:
    # rank = 7 * 4 = 28
    # Factor t = s * 4 + r where s in {0..6}, r in {0..3}
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # U rows indexed by A index = i1 * 4 + j1 * 2 + j2
    # V rows indexed by B index = j1 * 4 + j2 * 2 + k
    # W rows indexed by C index = k * n + i = k * 2 + i (per trace convention W[k*n+i, r])
    # Wait, the problem says W[k*n+i, r]. With n=2, that's W[2k+i, r].
    # C[i, k] corresponds to output element. The trace convention says W[k*n+i, r].
    # So W row index = k*2 + i where i is A row, k is B column.
    
    for s in range(7):  # Strassen factor
        for r in range(4):  # 1x2x2 factor
            t = s * 4 + r
            
            # U: for each A index
            for i1 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        a_idx = i1 * 4 + j1 * 2 + j2
                        U[a_idx][t] = U_strassen[i1][s] * U_122[j2][r]
            
            # V: for each B index
            for j1 in range(2):
                for j2 in range(2):
                    for k in range(2):
                        b_idx = j1 * 4 + j2 * 2 + k
                        V[b_idx][t] = V_strassen[j1][s] * V_122[j2 * 2 + k][r]
            
            # W: for each C index (trace convention: W[k*n+i, r])
            for i1 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        # C[i1, k1*2+k2], W index = k*2 + i where k = k1*2+k2, i = i1
                        w_idx = (k1 * 2 + k2) * 2 + i1
                        W[w_idx][t] = W_strassen[i1 * 2 + k1][s] * W_122[k2][r]
    
    return {"U": U, "V": V, "W": W}

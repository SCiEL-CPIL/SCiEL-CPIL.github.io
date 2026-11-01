# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 120.0
    n, m, p = 2, 2, 10
    num_blocks = p // m  # 5 blocks of 2x2
    
    # Rank-7 Strassen decomposition for 2x2 matrix multiplication
    # U: (n*m, 7) = (4, 7) - factors for A
    # V: (m*p_block, 7) = (2, 7) - factors for B (per block)
    # W: (p_block*n, 7) = (4, 7) - factors for C (per block)
    
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices per block: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C indices: (0,0)=c00, (0,1)=c01, (1,0)=c10, (1,1)=c11
    
    # U matrix (4x7): U[i*m+j, t] = coefficient of A[i,j] in product t
    U_block = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a: M1,M3,M5,M6
        [0, 0, 0, 0, 1, 0, 1],   # b: M5,M7
        [0, 1, 0, 0, 0, 1, 0],   # c: M2,M6
        [1, 1, 0, 1, 0, 0, -1]   # d: M1,M2,M4,M7
    ], dtype=float)
    
    # V matrix (4x7): V[j*p_block+k, t] = coefficient of B[j,k] in product t
    V_block = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # e: M1,M2,M4,M6
        [0, 0, 1, 0, 0, 1, 0],   # f: M3,M6
        [0, 0, -1, 1, 0, 0, 1],  # g: M3,M4,M7
        [1, 0, 0, 0, 1, 0, 1]    # h: M1,M5,M7
    ], dtype=float)
    
    # W matrix (4x7): W[(j)*n+i, t] = coefficient of C[i,j] in product t
    # C[0,0] = M1 - M5 + M4 + M7
    # C[0,1] = M5 + M3
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    W_block = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, -1, 1, 0, 0, 1, 0]    # C[1,1]
    ], dtype=float)
    
    # Verify the decomposition: sum_t U[:,t] * V[:,t] should give standard 2x2 mult
    # For each t, the outer product U_block[:,t] * V_block[:,t] contributes to C via W_block[:,t]
    # Check: for standard multiplication, C = A * B
    # The tensor is T[i,j,k] = delta[i,k] (A[i,j] * B[j,k] -> C[i,k])
    
    # Build full matrices
    rank = num_blocks * 7
    U_full = np.zeros((n * m, rank), dtype=float)
    V_full = np.zeros((m * p, rank), dtype=float)
    W_full = np.zeros((p * n, rank), dtype=float)
    
    for b in range(num_blocks):
        t_offset = b * 7
        
        # U is the same for all blocks (A is 2x2)
        U_full[:, t_offset:t_offset+7] = U_block
        
        # V: B has shape (2, 10), block b uses columns 2b, 2b+1
        V_full[:, t_offset:t_offset+7] = 0
        for jj in range(m):  # jj = 0, 1
            for kk in range(2):  # kk = 0, 1
                row_v = jj * p + (2 * b + kk)
                col_in_block = jj * 2 + kk
                V_full[row_v, t_offset:t_offset+7] = V_block[col_in_block, :]
        
        # W: C has shape (10, 2), block b uses rows 2b, 2b+1
        W_full[:, t_offset:t_offset+7] = 0
        for jj in range(2):  # jj = 0, 1 (column index in C, i.e., n-index)
            for ii in range(2):  # ii = 0, 1 (row index in C, i.e., p-index)
                row_w = (2 * b + ii) * n + jj
                col_in_block = ii * 2 + jj
                W_full[row_w, t_offset:t_offset+7] = W_block[col_in_block, :]
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Convert to integers where possible
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U_list]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V_list]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

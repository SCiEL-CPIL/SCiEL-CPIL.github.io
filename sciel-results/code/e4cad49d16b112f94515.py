import time
import numpy as np

DIMENSIONS = [3, 3, 8]

_START_TIME = time.time()
_TIME_LIMIT = 115.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 0] = -1; W[2, 0] = -1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[0, 5] = -1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[2, 6] = -1
    
    return U, V, W

def _construct_block_68():
    n, m, p = 3, 3, 8
    total_rank = 68
    
    U = np.zeros((n*m, total_rank)) # 9 x 68
    V = np.zeros((m*p, total_rank)) # 24 x 68
    W = np.zeros((p*n, total_rank)) # 24 x 68
    
    # Indices for A (9):
    # 0: 0,0; 1: 0,1; 2: 0,2
    # 3: 1,0; 4: 1,1; 5: 1,2
    # 6: 2,0; 7: 2,1; 8: 2,2
    
    # Indices for B (24):
    # j (row of B, 0..2), k (col of B, 0..7)
    # idx = j*8 + k
    
    # Indices for C (24):
    # k (col of C, 0..7), i (row of C, 0..2)
    # idx = k*3 + i
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:8]
    # A_sub is 2x2, B_sub is 2x8
    # Split B_sub into 4 blocks of 2x2: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    
    U_2x2, V_2x2, W_2x2 = _strassen_2x2()
    
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for l in range(7):
            # U: A[0:2, 0:2] -> indices 0, 1, 3, 4
            # U_2x2 indices: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
            # Map 2x2 A indices to 3x3 A indices:
            # 0 -> 0 (0,0)
            # 1 -> 1 (0,1)
            # 2 -> 3 (1,0)
            # 3 -> 4 (1,1)
            
            for u_idx in range(4):
                val = U_2x2[u_idx, l]
                if val != 0:
                    a_idx = [0, 1, 3, 4][u_idx]
                    U[a_idx, col] += val
            
            # V: B[0:2, k_start:k_start+2]
            # B indices: j (0,1), k (k_start, k_start+1)
            # V_2x2 indices: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
            # Map 2x2 B indices to 3x8 B indices:
            # 0 -> 0*8 + k_start
            # 1 -> 0*8 + k_start + 1
            # 2 -> 1*8 + k_start
            # 3 -> 1*8 + k_start + 1
            
            for v_idx in range(4):
                val = V_2x2[v_idx, l]
                if val != 0:
                    j, k_off = divmod(v_idx, 2)
                    b_idx = j * 8 + k_start + k_off
                    V[b_idx, col] += val
            
            # W: C[0:2, k_start:k_start+2]
            # C indices: i (0,1), k (k_start, k_start+1)
            # W_2x2 indices: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
            # Map 2x2 C indices to 3x8 C indices:
            # C idx = k*3 + i
            # 0 (i=0, k=0) -> k_start*3 + 0
            # 1 (i=0, k=1) -> (k_start+1)*3 + 0
            # 2 (i=1, k=0) -> k_start*3 + 1
            # 3 (i=1, k=1) -> (k_start+1)*3 + 1
            
            for w_idx in range(4):
                val = W_2x2[w_idx, l]
                if val != 0:
                    i, k_off = divmod(w_idx, 2)
                    c_idx = (k_start + k_off) * 3 + i
                    W[c_idx, col] += val
            
            col += 1
    
    # Block 2: A[0:2, 2:3] * B[2:3, 0:8]
    # A_sub is 2x1 (col 2 of A, rows 0,1). Indices: 2 (0,2), 5 (1,2)
    # B_sub is 1x8 (row 2 of B). Indices: 2*8 + k = 16+k
    # C_sub is 2x8 (rows 0,1 of C).
    # C[i,k] = A[i,2] * B[2,k]
    
    for i in range(2): # i=0,1
        a_idx = [2, 5][i]
        for k in range(8):
            b_idx = 16 + k
            c_idx = k * 3 + i
            
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
            
    # Block 3: A[2:3, 0:2] * B[0:2, 0:8]
    # A_sub is 1x2 (row 2 of A, cols 0,1). Indices: 6 (2,0), 7 (2,1)
    # B_sub is 2x8 (rows 0,1 of B).
    # C_sub is 1x8 (row 2 of C).
    # C[k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    
    for k in range(8):
        # Term 1: A[2,0] * B[0,k]
        U[6, col] = 1
        V[0*8 + k, col] = 1
        W[k*3 + 2, col] = 1
        col += 1
        
        # Term 2: A[2,1] * B[1,k]
        U[7, col] = 1
        V[1*8 + k, col] = 1
        W[k*3 + 2, col] = 1
        col += 1
        
    # Block 4: A[2:3, 2:3] * B[2:3, 0:8]
    # A_sub is 1x1 (A[2,2]). Index: 8
    # B_sub is 1x8 (row 2 of B). Indices: 16+k
    # C_sub is 1x8 (row 2 of C).
    # C[k] = A[2,2] * B[2,k]
    
    for k in range(8):
        U[8, col] = 1
        V[16 + k, col] = 1
        W[k*3 + 2, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(result, n, m, p):
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]
    
    # Construct tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    # Compute decomposition
    # T_comp[i, j, k] = sum_l U[i,l] V[j,l] W[k,l]
    # Use einsum
    T_comp = np.einsum('il,jl,kl->ijk', U, V, W)
    
    if not np.allclose(T, T_comp):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    result = _construct_block_68()
    
    # Verify
    if not _verify(result, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    return result

"""Matrix multiplication 6x6x6 rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 6, 6, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook rank 216
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def verify_fast(rank_val, U_mat, V_mat, W_mat):
        """Fast verification using numpy broadcasting."""
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape for efficient computation
        U_r = U_mat.reshape(n, m, rank_val)  # [i, j, c]
        V_r = V_mat.reshape(m, p, rank_val)  # [j, k, c]
        W_r = W_mat.reshape(p, n, rank_val)  # [k, i, c]
        
        # Compute tensor: T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # Use einsum: T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # All entries should be 1
        return np.all(T == 1)
    
    # Verify schoolbook
    assert verify_fast(rank_sb, U_sb, V_sb, W_sb), "Schoolbook verification failed"
    
    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2():
        r = 7
        U2 = np.zeros((4, r), dtype=np.int64)
        V2 = np.zeros((4, r), dtype=np.int64)
        W2 = np.zeros((4, r), dtype=np.int64)
        
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
        # M1 = (A00+A11)(B00+B11)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1  # C00 and C11
        
        # M2 = (A10+A11)B00
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1  # C10
        
        # M3 = A00(B10-B00)
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[0, 2] = -1
        W2[1, 2] = 1  # C01
        
        # M4 = A11(B11-B01)
        U2[3, 3] = 1
        V2[3, 3] = 1; V2[1, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1  # C00 and C10
        
        # M5 = (A00+A01)B11
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1  # -C00 + C01
        
        # M6 = (A10-A00)(B01+B11)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[1, 5] = 1; V2[3, 5] = 1
        W2[3, 5] = 1  # C11
        
        # M7 = (A01-A11)(B00+B10)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[0, 6] = 1; V2[2, 6] = 1
        W2[0, 6] = 1  # C00
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2()
    
    # Verify Strassen 2x2x2
    def verify_2x2():
        U_r = U2.reshape(2, 2, 7)
        V_r = V2.reshape(2, 2, 7)
        W_r = W2.reshape(2, 2, 7)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.all(T == 1)
    
    if not verify_2x2():
        # Strassen verification failed, return schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build 3x3x3 schoolbook (rank 27)
    r3 = 27
    U3 = np.zeros((9, r3), dtype=np.int64)
    V3 = np.zeros((9, r3), dtype=np.int64)
    W3 = np.zeros((9, r3), dtype=np.int64)
    
    col3 = 0
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                U3[i3*3 + j3, col3] = 1
                V3[j3*3 + k3, col3] = 1
                W3[k3*3 + i3, col3] = 1
                col3 += 1
    
    # Combine: total rank = 27 * 7 = 189
    total_rank = r3 * 7
    U_total = np.zeros((36, total_rank), dtype=np.int64)
    V_total = np.zeros((36, total_rank), dtype=np.int64)
    W_total = np.zeros((36, total_rank), dtype=np.int64)
    
    for c3 in range(r3):
        for c2 in range(7):
            c = c3 * 7 + c2
            for i in range(6):
                for j in range(6):
                    i2, i1 = divmod(i, 2)
                    j2, j1 = divmod(j, 2)
                    U_total[i*6+j, c] = U3[i2*3+j2, c3] * U2[i1*2+j1, c2]
            
            for j in range(6):
                for k in range(6):
                    j2, j1 = divmod(j, 2)
                    k2, k1 = divmod(k, 2)
                    V_total[j*6+k, c] = V3[j2*3+k2, c3] * V2[j1*2+k1, c2]
            
            for k in range(6):
                for i in range(6):
                    k2, k1 = divmod(k, 2)
                    i2, i1 = divmod(i, 2)
                    W_total[k*6+i, c] = W3[k2*3+i2, c3] * W2[k1*2+i1, c2]
    
    # Check time
    if time.time() > deadline:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Verify the 189-rank decomposition
    if verify_fast(total_rank, U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END

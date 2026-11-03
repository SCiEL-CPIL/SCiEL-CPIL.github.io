import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (16, rank)
    # W index: W[k*n + i, r]
    
    # Strassen decomposition for 2x2 matrix multiplication
    # C = A * B where A,B,C are 2x2
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]], C = [[c0, c1], [c2, c3]]
    # Indexing: A[i*2+j], B[i*2+j], C[i*2+j]
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1 = (a0+a3)(b0+b3) -> c0 += M1, c3 += M1
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    # M2 = (a2+a3)(b0) -> c2 += M2, c3 -= M2
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    # M3 = (a0)(b1-b3) -> c1 += M3, c3 += M3
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[3,2] = -1
    W_local[1,2] = 1; W_local[3,2] = 1
    
    # M4 = (a3)(b2-b0) -> c0 += M4, c2 += M4
    U_local[3,3] = 1
    V_local[2,3] = 1; V_local[0,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    # M5 = (a0+a1)(b3) -> c0 -= M5, c1 += M5
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    # M6 = (a2-a0)(b0+b1) -> c3 += M6
    U_local[2,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[3,5] = 1
    
    # M7 = (a1-a3)(b2+b3) -> c0 += M7
    U_local[1,6] = 1; U_local[3,6] = -1
    V_local[2,6] = 1; V_local[3,6] = 1
    W_local[0,6] = 1
    
    # Verify Strassen locally
    # Test with random 2x2 matrices
    A_test = np.array([[1,2],[3,4]], dtype=float)
    B_test = np.array([[5,6],[7,8]], dtype=float)
    C_test = A_test @ B_test  # [[19,22],[43,50]]
    
    # Flatten A, B, C
    A_flat = A_test.flatten()  # [a0,a1,a2,a3] = [1,2,3,4]
    B_flat = B_test.flatten()  # [b0,b1,b2,b3] = [5,6,7,8]
    
    # Compute via Strassen
    C_computed = np.zeros(4)
    for r in range(7):
        prod = np.dot(A_flat, U_local[:,r]) * np.dot(B_flat, V_local[:,r])
        C_computed += W_local[:,r] * prod
    
    assert np.allclose(C_computed, C_test.flatten()), f"Strassen failed: {C_computed} vs {C_test.flatten()}"
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # For each of the 8 sub-problems (j in 0..3, s in 0..1):
    # j selects which pair of rows of B (and columns of A)
    # s selects which pair of columns of B (and thus C)
    # 
    # A is 2x4: A[i, col] where col = 2*j + s_offset
    # For s=0: A uses columns 2*j, 2*j+1
    # For s=1: A uses columns 2*j+2, 2*j+3 ... wait, that's not right.
    
    # Let me reconsider the structure.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # Group by j: for j in {0,1} and j in {2,3}
    # C = sum_{j in {0,1}} A[i,j] B[j,k] + sum_{j in {2,3}} A[i,j] B[j,k]
    # 
    # For the first group: A_sub is 2x2 (columns 0,1), B_sub is 2x8 (rows 0,1)
    # For the second group: A_sub is 2x2 (columns 2,3), B_sub is 2x8 (rows 2,3)
    # 
    # Each 2x2 x 2x8 multiplication: view B_sub as 4 blocks of 2x2 (columns 0-1, 2-3, 4-5, 6-7)
    # Each 2x2 x 2x2 uses Strassen (rank 7)
    # So each 2x2 x 2x8 uses 4*7 = 28 rank
    # Total: 2 * 28 = 56
    
    for group in range(2):  # group 0: j in {0,1}, group 1: j in {2,3}
        for block_col in range(4):  # which 2-column block of B (and C)
            base_r = (group * 4 + block_col) * 7
            
            # A indices: A[i, 2*group + a_idx] where a_idx in {0,1}
            # U index: i*m + j = i*4 + (2*group + a_idx)
            # For Strassen local index la in {0,1,2,3} corresponding to A[la//2, la%2]
            a_map = [0]*4
            a_map[0] = 0*4 + (2*group + 0)  # A[0, 2*group]
            a_map[1] = 0*4 + (2*group + 1)  # A[0, 2*group+1]
            a_map[2] = 1*4 + (2*group + 0)  # A[1, 2*group]
            a_map[3] = 1*4 + (2*group + 1)  # A[1, 2*group+1]
            
            # B indices: B[j, k] where j = 2*group + b_row, k = 2*block_col + b_col
            # V index: j*p + k = (2*group + b_row)*8 + (2*block_col + b_col)
            # For Strassen local index lb in {0,1,2,3} corresponding to B[lb//2, lb%2]
            b_map = [0]*4
            b_map[0] = (2*group + 0)*8 + (2*block_col + 0)  # B[2*group, 2*block_col]
            b_map[1] = (2*group + 0)*8 + (2*block_col + 1)  # B[2*group, 2*block_col+1]
            b_map[2] = (2*group + 1)*8 + (2*block_col + 0)  # B[2*group+1, 2*block_col]
            b_map[3] = (2*group + 1)*8 + (2*block_col + 1)  # B[2*group+1, 2*block_col+1]
            
            # C indices: C[i, k] where k = 2*block_col + c_col
            # W index: k*n + i = (2*block_col + c_col)*2 + i
            # For Strassen local index lc in {0,1,2,3} corresponding to C[lc//2, lc%2]
            c_map = [0]*4
            c_map[0] = (2*block_col + 0)*2 + 0  # C[0, 2*block_col]
            c_map[1] = (2*block_col + 1)*2 + 0  # C[0, 2*block_col+1]
            c_map[2] = (2*block_col + 0)*2 + 1  # C[1, 2*block_col]
            c_map[3] = (2*block_col + 1)*2 + 1  # C[1, 2*block_col+1]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    # Verify the full decomposition with random matrices
    A_rand = np.random.randn(n, m)
    B_rand = np.random.randn(m, p)
    C_expected = A_rand @ B_rand
    
    # Compute via decomposition
    C_computed = np.zeros((n, p))
    for r in range(total_rank):
        u_vec = U[:, r]  # length 8
        v_vec = V[:, r]  # length 32
        w_vec = W[:, r]  # length 16
        
        # Compute linear forms
        a_form = 0.0
        for idx in range(n*m):
            i, j = divmod(idx, m)
            a_form += u_vec[idx] * A_rand[i, j]
        
        b_form = 0.0
        for idx in range(m*p):
            j, k = divmod(idx, p)
            b_form += v_vec[idx] * B_rand[j, k]
        
        prod = a_form * b_form
        
        for idx in range(p*n):
            k, i = divmod(idx, n)
            C_computed[i, k] += w_vec[idx] * prod
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback: return naive rank-64 decomposition
        U_naive = np.zeros((n*m, n*m*p), dtype=int)
        V_naive = np.zeros((m*p, n*m*p), dtype=int)
        W_naive = np.zeros((p*n, n*m*p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U_naive[i*m+j, r] = 1
                    V_naive[j*p+k, r] = 1
                    W_naive[k*n+i, r] = 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

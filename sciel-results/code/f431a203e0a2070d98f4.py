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
    # Indices: A is 2x2 (0,1,2,3), B is 2x2 (0,1,2,3), C is 2x2 (0,1,2,3)
    # U_local[i, r], V_local[j, r], W_local[k, r]
    # Note: In the local 2x2x2 problem:
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # But W index in global is k*n + i.
    # In local, if we map C indices 0,1,2,3 to k,i:
    # 0 -> k=0, i=0
    # 1 -> k=0, i=1
    # 2 -> k=1, i=0
    # 3 -> k=1, i=1
    # Wait, the standard Strassen formulas usually assume C is 2x2.
    # Let's stick to the verified local matrices from the current program.
    
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
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # We have 2 groups of j (0-1, 2-3) and 4 blocks of k (0-1, 2-3, 4-5, 6-7)
    # Total 8 sub-problems.
    for group in range(2):      # group 0: j in {0,1}, group 1: j in {2,3}
        for block_col in range(4):  # k-pair: {2*bc, 2*bc+1}
            base_r = (group * 4 + block_col) * 7
            
            # A indices: A[i, j] where j = 2*group + a_idx
            # a_idx 0,1,2,3 correspond to local indices 0,1,2,3
            # Local index mapping:
            # 0 -> (i=0, j_local=0) -> Global A[0, 2*group]
            # 1 -> (i=0, j_local=1) -> Global A[0, 2*group+1]
            # 2 -> (i=1, j_local=0) -> Global A[1, 2*group]
            # 3 -> (i=1, j_local=1) -> Global A[1, 2*group+1]
            a_map = [
                0*4 + (2*group + 0),
                0*4 + (2*group + 1),
                1*4 + (2*group + 0),
                1*4 + (2*group + 1)
            ]
            
            # B indices: B[j, k] where j = 2*group + b_row, k = 2*block_col + b_col
            # Local index mapping:
            # 0 -> (j_local=0, k_local=0) -> Global B[2*group, 2*block_col]
            # 1 -> (j_local=0, k_local=1) -> Global B[2*group, 2*block_col+1]
            # 2 -> (j_local=1, k_local=0) -> Global B[2*group+1, 2*block_col]
            # 3 -> (j_local=1, k_local=1) -> Global B[2*group+1, 2*block_col+1]
            b_map = [
                (2*group + 0)*8 + (2*block_col + 0),
                (2*group + 0)*8 + (2*block_col + 1),
                (2*group + 1)*8 + (2*block_col + 0),
                (2*group + 1)*8 + (2*block_col + 1)
            ]
            
            # C indices: C[i, k] where k = 2*block_col + c_col
            # W index: k*n + i
            # Local index mapping:
            # 0 -> (i=0, k_local=0) -> Global C[0, 2*block_col] -> W[(2*block_col)*2 + 0]
            # 1 -> (i=0, k_local=1) -> Global C[0, 2*block_col+1] -> W[(2*block_col+1)*2 + 0]
            # 2 -> (i=1, k_local=0) -> Global C[1, 2*block_col] -> W[(2*block_col)*2 + 1]
            # 3 -> (i=1, k_local=1) -> Global C[1, 2*block_col+1] -> W[(2*block_col+1)*2 + 1]
            c_map = [
                (2*block_col + 0)*2 + 0,
                (2*block_col + 1)*2 + 0,
                (2*block_col + 0)*2 + 1,
                (2*block_col + 1)*2 + 1
            ]
            
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
    
    # Verification
    A_test = np.random.randint(-5, 5, (n, m))
    B_test = np.random.randint(-5, 5, (m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p), dtype=int)
    for r in range(total_rank):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_vec = W[:, r]
        
        a_form = np.dot(A_test.flatten(), u_vec)
        b_form = np.dot(B_test.flatten(), v_vec)
        prod = a_form * b_form
        
        for idx in range(p * n):
            if w_vec[idx] != 0:
                k, i = divmod(idx, n)
                C_computed[i, k] += w_vec[idx] * prod
    
    if not np.allclose(C_computed, C_expected, atol=1e-6):
        # Fallback to naive rank-64 if something went wrong
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
    
    if time.time() - start_time > 140:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

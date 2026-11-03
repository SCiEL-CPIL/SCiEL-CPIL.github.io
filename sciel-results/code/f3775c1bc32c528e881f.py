import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Trivial rank-48 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
    
    def is_valid(U, V, W):
        """Check if U, V, W form a valid decomposition"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    # Construct Rank 42 Decomposition using Strassen's Algorithm
    # Decompose 2x4x6 into two 2x2x6 blocks
    # Each 2x2x6 is composed of three 2x2x2 blocks
    # Each 2x2x2 uses Strassen's rank 7 decomposition
    
    # Strassen rank 7 decomposition for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    # Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Matrix A (2x2): a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # Matrix B (2x2): e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # Matrix C (2x2): C[0,0]=M1-M4+M5-M7, C[0,1]=M3+M4, C[1,0]=M2+M5, C[1,1]=M1-M2+M3+M6
    
    # We need to construct U, V, W for the 2x2x2 tensor.
    # The tensor is sum of 7 terms.
    # Term r corresponds to multiplication M_r.
    # U_r is the coefficient vector for A in M_r.
    # V_r is the coefficient vector for B in M_r.
    # W_r is the coefficient vector for C in M_r.
    
    # U_r[i, j] is the coefficient of A[i, j] in M_r.
    # V_r[j, k] is the coefficient of B[j, k] in M_r.
    # W_r[k, i] is the coefficient of C[k, i] in M_r. (Note: C is p x n, here 2x2)
    
    # Let's define the 7 terms for 2x2x2:
    # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U1: A[0,0]=1, A[1,1]=1 -> [1, 0, 0, 1] (flattened i*m+j)
    # V1: B[0,0]=1, B[1,1]=1 -> [1, 0, 0, 1]
    # W1: C[0,0]+=1, C[1,1]+=1 -> [1, 0, 0, 1] (flattened k*n+i)
    # Note: C indices are (k, i). C[0,0] is k=0, i=0. C[1,1] is k=1, i=1.
    # C[0,1] is k=0, i=1. C[1,0] is k=1, i=0.
    
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    # U2: A[1,0]=1, A[1,1]=1 -> [0, 0, 1, 1]
    # V2: B[0,0]=1 -> [1, 0, 0, 0]
    # W2: C[1,0]-=1, C[1,1]-=1 -> [0, -1, -1, 0] (k=1, i=0 and k=1, i=1)
    # Wait, C[1,0] corresponds to k=1, i=0. Index in W: k*n+i = 1*2+0 = 2.
    # C[1,1] corresponds to k=1, i=1. Index in W: 1*2+1 = 3.
    # So W2[2] = -1, W2[3] = -1.
    
    # M3: A[0,0] * (B[0,1]-B[1,1])
    # U3: A[0,0]=1 -> [1, 0, 0, 0]
    # V3: B[0,1]=1, B[1,1]=-1 -> [0, 1, 0, -1]
    # W3: C[0,1]+=1, C[1,1]+=1 -> [0, 1, 0, 1]
    # C[0,1] is k=0, i=1. Index 1.
    # C[1,1] is k=1, i=1. Index 3.
    
    # M4: A[1,1] * (B[1,0]-B[0,0])
    # U4: A[1,1]=1 -> [0, 0, 0, 1]
    # V4: B[1,0]=1, B[0,0]=-1 -> [-1, 0, 1, 0]
    # W4: C[0,0]-=1, C[0,1]+=1 -> [-1, 1, 0, 0]
    # C[0,0] is k=0, i=0. Index 0.
    # C[0,1] is k=0, i=1. Index 1.
    
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    # U5: A[0,0]=1, A[0,1]=1 -> [1, 1, 0, 0]
    # V5: B[1,1]=1 -> [0, 0, 0, 1]
    # W5: C[0,0]+=1, C[1,0]+=1 -> [1, 0, 1, 0]
    # C[0,0] is k=0, i=0. Index 0.
    # C[1,0] is k=1, i=0. Index 2.
    
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # U6: A[1,0]=1, A[0,0]=-1 -> [-1, 0, 1, 0]
    # V6: B[0,0]=1, B[0,1]=1 -> [1, 1, 0, 0]
    # W6: C[1,1]+=1 -> [0, 0, 0, 1]
    # C[1,1] is k=1, i=1. Index 3.
    
    # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # U7: A[0,1]=1, A[1,1]=-1 -> [0, 1, 0, -1]
    # V7: B[1,0]=1, B[1,1]=1 -> [0, 0, 1, 1]
    # W7: C[0,0]-=1 -> [-1, 0, 0, 0]
    # C[0,0] is k=0, i=0. Index 0.

    # Let's build the 2x2x2 U, V, W matrices
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # Term 1
    U_222[:, 0] = [1, 0, 0, 1]
    V_222[:, 0] = [1, 0, 0, 1]
    W_222[:, 0] = [1, 0, 0, 1]
    
    # Term 2
    U_222[:, 1] = [0, 0, 1, 1]
    V_222[:, 1] = [1, 0, 0, 0]
    W_222[:, 1] = [0, 0, -1, -1]
    
    # Term 3
    U_222[:, 2] = [1, 0, 0, 0]
    V_222[:, 2] = [0, 1, 0, -1]
    W_222[:, 2] = [0, 1, 0, 1]
    
    # Term 4
    U_222[:, 3] = [0, 0, 0, 1]
    V_222[:, 3] = [-1, 0, 1, 0]
    W_222[:, 3] = [-1, 1, 0, 0]
    
    # Term 5
    U_222[:, 4] = [1, 1, 0, 0]
    V_222[:, 4] = [0, 0, 0, 1]
    W_222[:, 4] = [1, 0, 1, 0]
    
    # Term 6
    U_222[:, 5] = [-1, 0, 1, 0]
    V_222[:, 5] = [1, 1, 0, 0]
    W_222[:, 5] = [0, 0, 0, 1]
    
    # Term 7
    U_222[:, 6] = [0, 1, 0, -1]
    V_222[:, 6] = [0, 0, 1, 1]
    W_222[:, 6] = [-1, 0, 0, 0]

    # Now construct 2x4x6 decomposition
    # Split j (dim 4) into two groups of 2: j0 in {0,1}, j1 in {2,3}
    # Split k (dim 6) into three groups of 2: k0 in {0,1}, k1 in {2,3}, k2 in {4,5}
    # i is 0,1.
    
    # Total rank = 2 (for j split) * 3 (for k split) * 7 (Strassen) = 42
    
    rank_42 = 42
    U_42 = np.zeros((n * m, rank_42))
    V_42 = np.zeros((m * p, rank_42))
    W_42 = np.zeros((p * n, rank_42))
    
    r_offset = 0
    for j_block in range(2):
        # j indices in this block: j_block*2, j_block*2+1
        for k_block in range(3):
            # k indices in this block: k_block*2, k_block*2+1
            
            # Map Strassen U, V, W to the full U, V, W
            # U indices: i*m + j. i in {0,1}, j in {j_block*2, j_block*2+1}
            # V indices: j*p + k. j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
            # W indices: k*n + i. k in {k_block*2, k_block*2+1}, i in {0,1}
            
            for r in range(7):
                # U
                for i in range(2):
                    for jj in range(2):
                        j_idx = j_block * 2 + jj
                        u_val = U_222[i * 2 + jj, r]
                        if u_val != 0:
                            U_42[i * m + j_idx, r_offset + r] = u_val
                
                # V
                for jj in range(2):
                    for kk in range(2):
                        j_idx = j_block * 2 + jj
                        k_idx = k_block * 2 + kk
                        v_val = V_222[jj * 2 + kk, r]
                        if v_val != 0:
                            V_42[j_idx * p + k_idx, r_offset + r] = v_val
                
                # W
                for kk in range(2):
                    for i in range(2):
                        k_idx = k_block * 2 + kk
                        w_val = W_222[kk * 2 + i, r]
                        if w_val != 0:
                            W_42[k_idx * n + i, r_offset + r] = w_val
            
            r_offset += 7
    
    # Verify
    if is_valid(U_42, V_42, W_42):
        best_rank = 42
        best_U, best_V, best_W = U_42, V_42, W_42
    else:
        best_rank = rank_trivial
        best_U, best_V, best_W = U_trivial, V_trivial, W_trivial

    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END

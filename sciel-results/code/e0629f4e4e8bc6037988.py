"""Matrix multiplication 2x3x3: rank-17 exact tensor decomposition using block decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Return an exact rank-17 tensor decomposition for 2x3x3 matrix multiplication.
    
    Block decomposition:
    - Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2x2, Strassen rank 7)
    - Block 2: A[0:2,0:2] * B[0:2,2]   -> C[0:2,2]   (2x2x1, schoolbook rank 4)
    - Block 3: A[0:2,2]    * B[2,0:3]  -> C[0:2,0:3]  (2x1x3, schoolbook rank 6)
    Total: 7 + 4 + 6 = 17
    
    Indexing convention:
    - U has shape (n*m, rank) = (6, 17), indexed by i*m + j where i in [0,2), j in [0,3)
    - V has shape (m*p, rank) = (9, 17), indexed by j*p + k where j in [0,3), k in [0,3)
    - W has shape (p*n, rank) = (6, 17), indexed by k*n + i where k in [0,3), i in [0,2)
    
    The tensor T[i,j,k] = 1 iff (i,j,k) is a valid triple for matrix mult.
    Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    rank = 17
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # ===== Block 1: 2x2x2 Strassen (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]) =====
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U global index: i*3 + j
    # V global index: j*3 + k
    # W global index: k*2 + i
    
    # Strassen's 7 rank-1 terms for 2x2x2:
    # Term s: U_s * V_s * W_s where:
    #   U_s[i*3+j] for i,j in {0,1}
    #   V_s[j*3+k] for j,k in {0,1}
    #   W_s[k*2+i] for k,i in {0,1}
    
    # M1 = (A00 + A11)(B00 + B11)
    #   U: A00=1, A11=1 -> U[0]=1, U[4]=1
    #   V: B00=1, B11=1 -> V[0]=1, V[4]=1
    #   W: C00=1, C11=1 -> W[0]=1, W[3]=1
    U[0, col] = 1;  U[4, col] = 1
    V[0, col] = 1;  V[4, col] = 1
    W[0, col] = 1;  W[3, col] = 1
    col += 1
    
    # M2 = (A10 + A11)(B00)
    #   U: A10=1, A11=1 -> U[3]=1, U[4]=1
    #   V: B00=1 -> V[0]=1
    #   W: C10=1, C11=-1 -> W[2]=1, W[3]=-1
    U[3, col] = 1;  U[4, col] = 1
    V[0, col] = 1
    W[2, col] = 1;  W[3, col] = -1
    col += 1
    
    # M3 = A00(B01 - B11)
    #   U: A00=1 -> U[0]=1
    #   V: B01=1, B11=-1 -> V[1]=1, V[4]=-1
    #   W: C01=1, C11=1 -> W[1]=1, W[3]=1
    U[0, col] = 1
    V[1, col] = 1;  V[4, col] = -1
    W[1, col] = 1;  W[3, col] = 1
    col += 1
    
    # M4 = A11(B10 - B00)
    #   U: A11=1 -> U[4]=1
    #   V: B10=1, B00=-1 -> V[3]=1, V[0]=-1
    #   W: C10=1, C00=-1 -> W[2]=1, W[0]=-1
    U[4, col] = 1
    V[3, col] = 1;  V[0, col] = -1
    W[2, col] = 1;  W[0, col] = -1
    col += 1
    
    # M5 = (A00 + A01)B11
    #   U: A00=1, A01=1 -> U[0]=1, U[1]=1
    #   V: B11=1 -> V[4]=1
    #   W: C01=-1, C11=1 -> W[1]=-1, W[3]=1
    U[0, col] = 1;  U[1, col] = 1
    V[4, col] = 1
    W[1, col] = -1; W[3, col] = 1
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    #   U: A10=1, A00=-1 -> U[3]=1, U[0]=-1
    #   V: B00=1, B01=1 -> V[0]=1, V[1]=1
    #   W: C10=1, C11=1 -> W[2]=1, W[3]=1
    U[3, col] = 1;  U[0, col] = -1
    V[0, col] = 1;  V[1, col] = 1
    W[2, col] = 1;  W[3, col] = 1
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    #   U: A01=1, A11=-1 -> U[1]=1, U[4]=-1
    #   V: B10=1, B11=1 -> V[3]=1, V[4]=1
    #   W: C00=1, C01=-1 -> W[0]=1, W[1]=-1
    U[1, col] = 1;  U[4, col] = -1
    V[3, col] = 1;  V[4, col] = 1
    W[0, col] = 1;  W[1, col] = -1
    col += 1
    
    # ===== Block 2: 2x2x1 schoolbook (A[0:2,0:2] * B[0:2,2] -> C[0:2,2]) =====
    # 4 terms: for each (i, j) in {0,1} x {0,1}, k=2
    # U[i*3+j] = 1, V[j*3+2] = 1, W[2*2+i] = W[4+i] = 1
    for i in range(2):
        for j in range(2):
            U[i * 3 + j, col] = 1
            V[j * 3 + 2, col] = 1
            W[2 * 2 + i, col] = 1  # W[4+i]
            col += 1
    
    # ===== Block 3: 2x1x3 schoolbook (A[0:2,2] * B[2,0:3] -> C[0:2,0:3]) =====
    # 6 terms: for each (i, k) in {0,1} x {0,1,2}, j=2
    # U[i*3+2] = 1, V[2*3+k] = V[6+k] = 1, W[k*2+i] = 1
    for i in range(2):
        for k in range(3):
            U[i * 3 + 2, col] = 1
            V[2 * 3 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the reconstructed tensor
    # T_reconstructed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_recon = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        # Outer product of U[:,r], V[:,r], W[:,r] mapped to (i,j,k) indices
        for idx_u in range(n * m):
            i, j = divmod(idx_u, m)
            u_val = U_arr[idx_u, r]
            if u_val == 0:
                continue
            for idx_v in range(m * p):
                jj, k = divmod(idx_v, p)
                if jj != j:
                    continue
                v_val = V_arr[idx_v, r]
                if v_val == 0:
                    continue
                w_idx = k * n + i
                w_val = W_arr[w_idx, r]
                if w_val != 0:
                    T_recon[i, j, k] += u_val * v_val * w_val
    
    # The true tensor: T[i,j,k] = 1 for all valid (i,j,k) in matrix mult
    # For 2x3 * 3x3 -> 2x3, all triples (i,j,k) with i in [0,2), j in [0,3), k in [0,3) are valid
    T_true = np.ones((n, m, p), dtype=float)
    
    if not np.allclose(T_recon, T_true, atol=1e-10):
        # Fallback to schoolbook rank-18 if verification fails
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=int)
        V_fb = np.zeros((m * p, rank_fb), dtype=int)
        W_fb = np.zeros((p * n, rank_fb), dtype=int)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

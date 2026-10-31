import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    # Define the 7-term Strassen decomposition for 2x2 matrices
    # U2: shape (4, 7), index [i*2+j, t]
    # V2: shape (4, 7), index [j*2+k, t]
    # W2: shape (4, 7), index [k*2+i, t] (Trace convention: k*n+i, here n=2 so k*2+i)
    
    # Coefficients for A (U2)
    # t=0: A00+A11 -> [1,0,0,1]
    # t=1: A10+A11 -> [0,0,1,1]
    # t=2: A00      -> [1,0,0,0]
    # t=3: A11      -> [0,0,0,1]
    # t=4: A00+A01  -> [1,1,0,0]
    # t=5: A10-A00  -> [-1,0,1,0]
    # t=6: A01-A11  -> [0,1,0,-1]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ], dtype=int)
    
    # Coefficients for B (V2)
    # t=0: B00+B11 -> [1,0,0,1]
    # t=1: B00     -> [1,0,0,0]
    # t=2: B01-B11 -> [0,1,-1,0]
    # t=3: B10-B00 -> [-1,0,1,0]
    # t=4: B11     -> [0,0,0,1]
    # t=5: B00+B01 -> [1,1,0,0]
    # t=6: B10+B11 -> [0,0,1,1]
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ], dtype=int)
    
    # Coefficients for C (W2)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    # Indices for W2 are [k*2+i, t]
    # k=0, i=0 (C00): [1, 0, 0, 1, -1, 0, 1]
    # k=1, i=0 (C10): [0, 1, 0, 1, 0, 0, 0]
    # k=0, i=1 (C01): [0, 0, 1, 0, 1, 0, 0]
    # k=1, i=1 (C11): [1, -1, 1, 0, 0, 1, 0]
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=1, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=0, i=1)
        [1, -1, 1, 0, 0, 1, 0],  # C11 (k=1, i=1)
    ])
                                 # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r]?
                                 # Let's check the current code's W index usage.
                                 # Current code: W[k * 4 + i][c].
                                 # This matches W[k, i] with n=4.
                                 # In the 2x2 sub-problem, it should be W[k, i] with n=2.
                                 # So W2 index is k*2 + i.
                                 # C00 is i=0, k=0. Index 0*2+0=0.
                                 # C01 is i=0, k=1. Index 1*2+0=2.
                                 # C10 is i=1, k=0. Index 0*2+1=1.
                                 # C11 is i=1, k=1. Index 1*2+1=3.
                                 # Let's re-verify W2 values based on C[i][k].
                                 # C[0][0] = M0 + M3 - M4 + M6. 
                                 #   M0:1, M1:0, M2:0, M3:1, M4:-1, M5:0, M6:1.
                                 # C[1][0] = M1 + M3.
                                 #   M0:0, M1:1, M2:0, M3:1, M4:0, M5:0, M6:0.
                                 # C[0][1] = M2 + M4.
                                 #   M0:0, M1:0, M2:1, M3:0, M4:1, M5:0, M6:0.
                                 # C[1][1] = M0 - M1 + M2 + M5.
                                 #   M0:1, M1:-1, M2:1, M3:0, M4:0, M5:1, M6:0.
                                 #
                                 # W2 rows (index k*2+i):
                                 # Row 0 (k=0, i=0 -> C00): [1, 0, 0, 1, -1, 0, 1]
                                 # Row 1 (k=0, i=1 -> C10): [0, 1, 0, 1, 0, 0, 0]
                                 # Row 2 (k=1, i=0 -> C01): [0, 0, 1, 0, 1, 0, 0]
                                 # Row 3 (k=1, i=1 -> C11): [1, -1, 1, 0, 0, 1, 0]
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1 (C11)
    ], dtype=int)

    # We need to construct U, V, W of shape (16, 49)
    # U[i*4+j, s*7+t] = U2[I*2+J, s] * U2[i2*2+j2, t] ?
    # Let's derive the mapping carefully.
    # A is 4x4. Elements A[r, c].
    # Block indices: I = r // 2, J = c // 2.
    # Local indices: i2 = r % 2, j2 = c % 2.
    # The coefficient of A[r, c] in the (s, t)-th multiplication is:
    #   coeff_A_block[I, J, s] * coeff_A_local[i2, j2, t]
    #   where coeff_A_block is from the "outer" Strassen on 2x2 blocks of 2x2 matrices?
    #   No, the "outer" Strassen operates on the blocks A00, A01, A10, A11.
    #   The term is M_{s,t} = (sum_{I,J} beta_{s,I,J} A_{I,J}) * (sum_{J,K} gamma_{s,J,K} B_{J,K}) ...
    #   Wait, the standard tensor product of decompositions works like this:
    #   T_4 = T_2 \otimes T_2
    #   U_4 = U_2 \otimes U_2
    #   V_4 = V_2 \otimes V_2
    #   W_4 = W_2 \otimes W_2
    #   But we must respect the index order.
    
    # Let's use broadcasting to create the full matrices.
    # U2 has shape (4, 7). Rows are (I, i2) flattened as I*2+i2? No, standard C-order is I*2+i2 if we view it as 2x2.
    # In our U2 definition above, row 0 is A00 (I=0, i2=0, j2=0)?
    # My U2 array was:
    # Row 0: A00 -> I=0, j2=0? No, A is [I][J].
    # Let's re-map U2 rows to (I, J) blocks? No, U2 is for the 2x2 matrix A.
    # The 2x2 matrix A has elements A00, A01, A10, A11.
    # In the 4x4 decomposition, A is composed of 2x2 blocks.
    # Let A_block[I][J] be the 2x2 submatrix.
    # The element A[r][c] is A_block[I][J][i2][j2] where I=r//2, J=c//2.
    # The coefficient of A[r][c] in the s-th outer multiplication is beta[s][I][J].
    # The coefficient of the local element A_block[I][J][i2][j2] in the t-th inner multiplication is U2[i2*2+j2][t].
    # So U_full[r*4+c, s*7+t] = beta[s][I][J] * U2[i2*2+j2][t].
    
    # We need beta, gamma, delta matrices of shape (7, 2, 2).
    # beta[s, I, J]
    # gamma[s, J, K]
    # delta[s, I, K]
    
    # From previous code:
    # s=0: beta[0,0,0]=1, beta[0,1,1]=1; gamma[0,0,0]=1, gamma[0,1,1]=1; delta[0,0,0]=1, delta[0,1,1]=1
    # s=1: beta[1,1,0]=1, beta[1,1,1]=1; gamma[1,0,0]=1; delta[1,1,0]=1, delta[1,1,1]=-1
    # s=2: beta[2,0,0]=1; gamma[2,0,1]=1, gamma[2,1,1]=-1; delta[2,0,1]=1, delta[2,1,1]=1
    # s=3: beta[3,1,1]=1; gamma[3,1,0]=1, gamma[3,0,0]=-1; delta[3,0,0]=1, delta[3,1,0]=1
    # s=4: beta[4,0,0]=1, beta[4,0,1]=1; gamma[4,1,1]=1; delta[4,0,0]=-1, delta[4,0,1]=1
    # s=5: beta[5,1,0]=1, beta[5,0,0]=-1; gamma[5,0,0]=1, gamma[5,0,1]=1; delta[5,1,1]=1
    # s=6: beta[6,0,1]=1, beta[6,1,1]=-1; gamma[6,1,0]=1, gamma[6,1,1]=1; delta[6,0,0]=1
    
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)
    
    beta[0,0,0]=1; beta[0,1,1]=1
    gamma[0,0,0]=1; gamma[0,1,1]=1
    delta[0,0,0]=1; delta[0,1,1]=1
    
    beta[1,1,0]=1; beta[1,1,1]=1
    gamma[1,0,0]=1
    delta[1,1,0]=1; delta[1,1,1]=-1
    
    beta[2,0,0]=1
    gamma[2,0,1]=1; gamma[2,1,1]=-1
    delta[2,0,1]=1; delta[2,1,1]=1
    
    beta[3,1,1]=1
    gamma[3,1,0]=1; gamma[3,0,0]=-1
    delta[3,0,0]=1; delta[3,1,0]=1
    
    beta[4,0,0]=1; beta[4,0,1]=1
    gamma[4,1,1]=1
    delta[4,0,0]=-1; delta[4,0,1]=1
    
    beta[5,1,0]=1; beta[5,0,0]=-1
    gamma[5,0,0]=1; gamma[5,0,1]=1
    delta[5,1,1]=1
    
    beta[6,0,1]=1; beta[6,1,1]=-1
    gamma[6,1,0]=1; gamma[6,1,1]=1
    delta[6,0,0]=1

    # Construct U (16, 49)
    # U[r*4+c, s*7+t] = beta[s, I, J] * U2[i2*2+j2, t]
    # r = 2*I + i2, c = 2*J + j2
    # I in [0,1], J in [0,1], i2 in [0,1], j2 in [0,1]
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for I in range(2):
        for J in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    r = 2*I + i2
                    c = 2*J + j2
                    row_u = r*4 + c
                    # U2 index: i2*2 + j2
                    u2_idx = i2*2 + j2
                    # For all s, t:
                    # U[row_u, s*7+t] += beta[s, I, J] * U2[u2_idx, t]
                    # This is a rank-1 update or can be done via broadcasting.
                    # beta[:, I, J] is shape (7,)
                    # U2[u2_idx, :] is shape (7,)
                    # We need to fill 49 columns.
                    # col = s*7 + t.
                    # We can reshape beta[:, I, J] to (7, 1) and U2[u2_idx, :] to (1, 7) and outer product?
                    # Yes.
                    outer_u = np.outer(beta[:, I, J], U2[u2_idx, :])
                    U[row_u, :] += outer_u.flatten()
                    
                    # Similarly for V
                    # V[c*4+k, s*7+t] = gamma[s, J, K] * V2[j2*2+k2, t]
                    # V row index: c*4 + k.
                    # c = 2*J + j2.
                    # We loop over K, k2.
        
    # Let's do V and W similarly in nested loops but using NumPy outer for speed/clarity change.
    
    for J in range(2):
        for K in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    c = 2*J + j2
                    k = 2*K + k2
                    row_v = c*4 + k
                    v2_idx = j2*2 + k2
                    outer_v = np.outer(gamma[:, J, K], V2[v2_idx, :])
                    V[row_v, :] += outer_v.flatten()
                    
    for I in range(2):
        for K in range(2):
            for i2 in range(2):
                for k2 in range(2):
                    k = 2*K + k2
                    i = 2*I + i2
                    row_w = k*4 + i # Trace convention W[k*n+i]
                    w2_idx = k2*2 + i2
                    outer_w = np.outer(delta[:, I, K], W2[w2_idx, :])
                    W[row_w, :] += outer_w.flatten()

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

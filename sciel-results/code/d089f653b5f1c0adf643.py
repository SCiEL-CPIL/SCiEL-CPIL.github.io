import time
import sys
import numpy as np
import scipy.linalg

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Local indices:
    # A: 2x2 -> flat i*2+j
    # B: 2x2 -> flat j*2+k
    # C: 2x2 -> flat i*2+k (local), mapped to W convention k*n+i globally
    
    # Strassen components:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11
    
    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11
    
    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11
    
    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10
    
    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01
    
    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11
    
    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [
                0 * 4 + (2 * s + 0),
                0 * 4 + (2 * s + 1),
                1 * 4 + (2 * s + 0),
                1 * 4 + (2 * s + 1),
            ]
            
            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),
                (2 * s + 0) * 8 + (2 * j + 1),
                (2 * s + 1) * 8 + (2 * j + 0),
                (2 * s + 1) * 8 + (2 * j + 1),
            ]
            
            c_map = [
                (2 * j + 0) * 2 + 0,
                (2 * j + 1) * 2 + 0,
                (2 * j + 0) * 2 + 1,
                (2 * j + 1) * 2 + 1,
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val
                        
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build tensor matrix: M[i,j,k, r] = U[i,r] * V[j,r] * W[k,r]
    # Shape: (n*m*p) x total_rank = 64 x 56
    M = np.zeros((n * m * p, total_rank))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # Find null space to reduce rank
    try:
        # Use QR with column pivoting to find a linearly independent subset of columns
        # This is more robust than SVD + greedy elimination
        Q, R, P = scipy.linalg.qr(M, pivoting=True)
        
        # Determine numerical rank
        # The diagonal of R indicates the rank. 
        # We look for the last significant diagonal element.
        tol = 1e-10
        diag_R = np.abs(np.diag(R))
        
        # Find the rank
        rank_m = 0
        for i in range(len(diag_R)):
            if diag_R[i] > tol:
                rank_m = i + 1
            else:
                break
        
        # If rank reduction is possible
        if rank_m < total_rank:
            # P is the permutation vector. 
            # The first rank_m columns of M[:, P] form a basis.
            # We want to keep the original columns corresponding to P[0:rank_m]
            keep_indices = P[:rank_m]
            
            U_new = U_np[:, keep_indices]
            V_new = V_np[:, keep_indices]
            W_new = W_np[:, keep_indices]
            
            new_rank = len(keep_indices)
            
            # Verify correctness
            # Reconstruct tensor and check against expected (all ones)
            tensor_recon = np.zeros((n, m, p))
            for r in range(new_rank):
                # u_r, v_r, w_r
                u_r = U_new[:, r]
                v_r = V_new[:, r]
                w_r = W_new[:, r]
                tensor_recon += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
            
            # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
            max_err = np.max(np.abs(tensor_recon - 1.0))
            
            if max_err < 1e-8:
                U = U_new.tolist()
                V = V_new.tolist()
                W = W_new.tolist()
    except Exception:
        pass
    
    if time.time() - start_time > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

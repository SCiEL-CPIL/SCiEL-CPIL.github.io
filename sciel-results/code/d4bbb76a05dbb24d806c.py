import time
import sys
import numpy as np
import scipy.linalg

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8

    # Strategy: Use Strassen on 2x2 blocks to get rank 56, then reduce via QR.
    # Theoretical rank of 2x2x8 is 24, so total rank 48 is possible.
    
    # Strassen decomposition for 2x2 (rank 7)
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1; W_local[3][0] = 1
    
    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1; W_local[3][1] = -1
    
    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1; W_local[3][2] = 1
    
    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1; W_local[2][3] = 1
    
    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1; W_local[1][4] = 1
    
    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Decompose into 8 blocks of 2x2 x 2x2
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
                        
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build tensor matrix: M[i*m*p + j*p + k, r] = U[i,r] * V[j,r] * W[k,r]
    # Note: The tensor T is defined such that T[i,j,k] = 1 for all i,j,k.
    # The decomposition is sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k.
    M = np.zeros((n * m * p, total_rank))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # Find null space to reduce rank
    try:
        # Use SVD for better numerical stability in finding the rank
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        
        tol = 1e-9
        rank_m = np.sum(s > tol)
        
        if rank_m < total_rank:
            # The first rank_m columns of Vt correspond to the significant components
            # But we need the basis for the column space of M
            # The columns of M are linear combinations of the rank-1 tensors.
            # We want to find a subset of the original rank-1 tensors that spans the same space.
            # Actually, SVD gives us a new basis. We need to map back to the original factors.
            # This is tricky. The standard way is to find the null space of the matrix of coefficients.
            
            # Let's stick to QR with pivoting as in the original, but ensure it's robust.
            Q, R, P = scipy.linalg.qr(M, pivoting=True)
            
            diag_R = np.abs(np.diag(R))
            
            rank_m = 0
            for i in range(len(diag_R)):
                if diag_R[i] > tol:
                    rank_m = i + 1
                else:
                    break
            
            if rank_m < total_rank:
                keep_indices = P[:rank_m]
                
                U_new = U_np[:, keep_indices]
                V_new = V_np[:, keep_indices]
                W_new = W_np[:, keep_indices]
                
                new_rank = len(keep_indices)
                
                # Verify correctness
                tensor_recon = np.zeros((n, m, p))
                for r in range(new_rank):
                    u_r = U_new[:, r]
                    v_r = V_new[:, r]
                    w_r = W_new[:, r]
                    tensor_recon += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
                
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

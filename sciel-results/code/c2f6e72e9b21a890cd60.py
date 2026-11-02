import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 140
    n, m, p = DIMENSIONS
    
    # Strassen vectors for 2x2 matrix multiplication (rank 7)
    u_vecs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    v_vecs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    w_vecs = [
        [1, 0, 0, 1],   # M1: c00 + c11
        [0, 0, 1, -1],  # M2: c10 - c11
        [0, 1, 0, 1],   # M3: c01 + c11
        [1, 0, 1, 0],   # M4: c00 + c10
        [-1, 1, 0, 0],  # M5: -c00 + c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ]
    
    def build_strassen():
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(8):
            base_col = 7 * r
            for s in range(7):
                col = base_col + s
                
                for li in range(4):
                    if u_vecs[s][li] != 0:
                        U[li, col] = u_vecs[s][li]
                        
                v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
                for li in range(4):
                    if v_vecs[s][li] != 0:
                        V[v_idx[li], col] = v_vecs[s][li]
                        
                w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
                for li in range(4):
                    if w_vecs[s][li] != 0:
                        W[w_idx[li], col] = w_vecs[s][li]
                        
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        # U: (2, 2, rank)
        # V: (2, 16, rank)
        # W: (16, 2, rank)
        U_arr = U.reshape(n, m, -1)
        V_arr = V.reshape(m, p, -1)
        W_arr = W.reshape(p, n, -1)
        
        # Contract over rank
        # T[i, j, k, i'] = sum_r U[i,j,r] V[j,k,r] W[k,i',r]
        # We want T[i, j, k, i'] == delta(i, i')
        
        # Use einsum: 'ijr,jkr,ki'r -> ijk i'? No.
        # U: i j r
        # V: j k r
        # W: k i' r
        # Result: i j k i'
        T = np.einsum('ijr,jkr,kiqr->ijkq', U_arr, V_arr, W_arr)
        
        # Check if T[i, j, k, i'] is 1 if i==i' else 0
        # i ranges 0..1, i' ranges 0..1
        # j, k are free
        valid = True
        for i in range(n):
            for i_prime in range(n):
                expected = 1.0 if i == i_prime else 0.0
                # T[:, :, :, i_prime] should be expected * Identity(j, k)?
                # No, T[i, j, k, i_prime] should be expected for ALL j, k?
                # Let's check schoolbook.
                # U[i,j,r]=1, V[j,k,r]=1, W[k,i',r]=1.
                # T[i,j,k,i'] = 1 if i=i'.
                # So yes, for fixed i, i', T[i, :, :, i'] should be all expected.
                block = T[i, :, :, i_prime]
                if not np.allclose(block, expected, atol=tol):
                    valid = False
                    break
            if not valid:
                break
        return valid

    U, V, W = build_strassen()
    
    if verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+l, col] = 1
                W_fb[l*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END

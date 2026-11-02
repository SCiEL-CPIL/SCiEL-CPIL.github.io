import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 130
    
    # Winograd 2x2x2 decomposition (Rank 7)
    # A: (a,b,c,d) -> (0,1,2,3)
    # B: (e,f,g,h) -> (0,1,2,3)
    # C: (p,q,r,s) -> (0,1,2,3)
    # Terms:
    # 1: (a+d)(e+h) -> C: p+1, s+1
    # 2: (c+d)e    -> C: r+1, s-1
    # 3: a(f-h)    -> C: q+1
    # 4: d(g-e)    -> C: p-1, q+1, r+1
    # 5: (a+b)g    -> C: p+1, q+1
    # 6: (c-a)(e+f)-> C: r-1, s+1
    # 7: (b-d)(h+f)-> C: p-1, q+1, r+1
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ], dtype=np.float64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 1],  # f
        [0, 0, 0, 1, 1, 0, 0],  # g
        [1, 0, -1, 0, 0, 0, 1]  # h
    ], dtype=np.float64)
    
    W2 = np.array([
        [1, 0, 0, -1, 1, 0, -1], # p
        [0, 0, 1, 1, 1, 0, 1],   # q
        [0, 1, 0, 1, 0, -1, 1],  # r
        [1, -1, 0, 0, 0, 1, 0]   # s
    ], dtype=np.float64)
    
    total_rank = 108
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Part 1: j=0 (4x1x6), Rank 24
    # A: i in 0..3, j=0. Index: i*5 + 0
    # B: j=0, k in 0..5. Index: 0*6 + k
    # C: k in 0..5, i in 0..3. Index: k*4 + i
    
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1
            
    # Part 2: j=1..4 (4x4x6), Rank 84
    # Split i into (0,1) and (2,3)
    # Split k into (0,1), (2,3), (4,5)
    
    i_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for i_start, i_end in i_blocks:
        for k_start, k_end in k_blocks:
            # This block is 2x4x2
            # A: rows i_start, i_start+1. Cols j=1..4.
            # B: rows j=1..4. Cols k_start, k_start+1.
            # C: rows i_start, i_start+1. Cols k_start, k_start+1.
            
            # Split middle dimension (j=1..4) into j=1,2 and j=3,4
            # Sub-block 1: j=1,2
            # Sub-block 2: j=3,4
            
            for j_offset in [0, 2]:
                # j indices in global B/A: 1+j_offset, 2+j_offset
                # Local 2x2x2 block
                
                # A indices (2x2):
                # (i_start, 1+j_offset), (i_start, 2+j_offset)
                # (i_start+1, 1+j_offset), (i_start+1, 2+j_offset)
                a_idx = [
                    i_start * m + (1 + j_offset),
                    i_start * m + (2 + j_offset),
                    (i_start + 1) * m + (1 + j_offset),
                    (i_start + 1) * m + (2 + j_offset)
                ]
                
                # B indices (2x2):
                # (1+j_offset, k_start), (1+j_offset, k_start+1)
                # (2+j_offset, k_start), (2+j_offset, k_start+1)
                b_idx = [
                    (1 + j_offset) * p + k_start,
                    (1 + j_offset) * p + (k_start + 1),
                    (2 + j_offset) * p + k_start,
                    (2 + j_offset) * p + (k_start + 1)
                ]
                
                # C indices (2x2):
                # (k_start, i_start), (k_start, i_start+1)
                # (k_start+1, i_start), (k_start+1, i_start+1)
                c_idx = [
                    k_start * n + i_start,
                    k_start * n + (i_start + 1),
                    (k_start + 1) * n + i_start,
                    (k_start + 1) * n + (i_start + 1)
                ]
                
                # Add Winograd 2x2x2
                for r in range(7):
                    for li in range(4):
                        U[a_idx[li], col + r] += U2[li, r]
                        V[b_idx[li], col + r] += V2[li, r]
                        W[c_idx[li], col + r] += W2[li, r]
                col += 7
                
    # Verify
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook
        U, V, W = np.zeros((n*m, n*m*p)), np.zeros((m*p, n*m*p)), np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        # Trim
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    else:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

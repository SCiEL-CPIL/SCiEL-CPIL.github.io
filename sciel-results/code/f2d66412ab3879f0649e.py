import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Local 2x2 Strassen vectors
    u_local = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_local = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_local = [
        [1, 0, 0, 1],
        [0, 0, 0, -1],
        [0, 0, 1, 1],
        [-1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0]
    ]
    
    # Block 2x2 Strassen matrices
    U_blocks = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    V_blocks = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    W_blocks = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [0, -1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[-1, 0], [0, 0]])
    ]
    
    col = 0
    for offset in [0, 4]: # 0 for B_L, 4 for B_R
        for s in range(7):
            for c in range(7):
                # Construct global vectors
                u_g = np.kron(U_blocks[s], u_local[c])
                v_g = np.kron(V_blocks[s], v_local[c])
                w_g = np.kron(W_blocks[s], w_local[c])
                
                # Flatten and assign
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                
                # u_g is 4x4 flattened to 16
                # v_g is 4x4 flattened to 16, but needs to map to 32 (j, k)
                # w_g is 4x4 flattened to 16, but needs to map to 32 (k, i)
                
                # Map u_g to U
                # u_g[i_block, j_block, i_local, j_local]
                # Global i = 2*i_block + i_local
                # Global j = 2*j_block + j_local
                U_flat = np.zeros(16)
                for ib in range(2):
                    for jb in range(2):
                        for il in range(2):
                            for jl in range(2):
                                gi = 2*ib + il
                                gj = 2*jb + jl
                                U_flat[gi*m + gj] = U_blocks[s, ib, jb] * u_local[c, il*2+il] # Wait, u_local is flattened 4x4?
                                # u_local[c] is length 4. Index il*2+il? No.
                                # u_local[c] is [u00, u01, u10, u11].
                                # So index is il*2 + jl.
                                U_flat[gi*m + gj] = U_blocks[s, ib, jb] * u_local[c][il*2 + jl]
                U[:, col] = U_flat
                
                # Map v_g to V
                # v_g[j_block, k_block, j_local, k_local]
                # Global j = 2*j_block + j_local
                # Global k = offset + 2*k_block + k_local
                V_flat = np.zeros(32)
                for jb in range(2):
                    for kb in range(2):
                        for jl in range(2):
                            for kl in range(2):
                                gj = 2*jb + jl
                                gk = offset + 2*kb + kl
                                V_flat[gj*p + gk] = V_blocks[s, jb, kb] * v_local[c][jl*2 + kl]
                V[:, col] = V_flat
                
                # Map w_g to W
                # w_g[k_block, i_block, k_local, i_local]
                # Global k = offset + 2*k_block + k_local
                # Global i = 2*i_block + i_local
                W_flat = np.zeros(32)
                for kb in range(2):
                    for ib in range(2):
                        for kl in range(2):
                            for il in range(2):
                                gk = offset + 2*kb + kl
                                gi = 2*ib + il
                                W_flat[gk*n + gi] = W_blocks[s, kb, ib] * w_local[c][kl*2 + il]
                W[:, col] = W_flat
                
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

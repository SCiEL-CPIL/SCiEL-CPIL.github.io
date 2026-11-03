import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        return np.max(np.abs(T_computed - T_expected)) < tol

    def build_block_diagonal():
        """
        Block-diagonal embedding:
        - Partition i into {0,1} and {2}
        - For i in {0,1}: partition j into {0,1} and {2}
          - i in {0,1}, j in {0,1}: 2x2x8 tensor, use 4 Strassen blocks (rank 28)
          - i in {0,1}, j in {2}: 2x1x8 tensor, schoolbook (rank 16)
        - For i in {2}: 1x3x8 tensor, schoolbook (rank 24)
        Total rank: 28 + 16 + 24 = 68
        """
        rank_total = 68
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        col = 0
        
        # Block 1: i in {0,1}, j in {0,1}, k in {0,...,7}
        # Use 4 Strassen blocks for k = {0,1}, {2,3}, {4,5}, {6,7}
        strassen_U = [
            {0: 1, 3: 1},  # (0,0), (1,1)
            {2: 1, 3: 1},  # (1,0), (1,1)
            {0: 1},         # (0,0)
            {3: 1},         # (1,1)
            {0: 1, 1: 1},   # (0,0), (0,1)
            {2: 1, 0: -1},  # (1,0), (0,0)
            {1: 1, 3: -1}   # (0,1), (1,1)
        ]
        strassen_V = [
            {0: 1, 3: 1},  # (0,0), (1,1)
            {0: 1},         # (0,0)
            {1: 1, 3: -1},  # (0,1), (1,1)
            {2: 1, 0: -1},  # (1,0), (0,0)
            {3: 1},         # (1,1)
            {0: 1, 1: 1},   # (0,0), (0,1)
            {2: 1, 3: 1}    # (1,0), (1,1)
        ]
        strassen_W = [
            {0: 1, 3: 1},  # (0,0), (1,1)
            {2: 1, 3: -1},  # (1,0), (1,1)
            {1: 1, 3: 1},   # (0,1), (1,1)
            {0: 1, 2: 1},   # (0,0), (1,0)
            {0: -1, 1: 1},  # (0,0), (0,1)
            {3: 1},         # (1,1)
            {0: 1}          # (0,0)
        ]
        
        for g in range(4):
            k0 = 2 * g
            for l in range(7):
                # U: i in {0,1}, j in {0,1}
                for li, coeff in strassen_U[l].items():
                    i_local = li // 2
                    j_local = li % 2
                    U[i_local*3 + j_local, col] += coeff
                
                # V: j in {0,1}, k in {k0, k0+1}
                for lj, coeff in strassen_V[l].items():
                    j_local = lj // 2
                    k_local = lj % 2
                    k_global = k0 + k_local
                    V[j_local*8 + k_global, col] += coeff
                
                # W: k in {k0, k0+1}, i in {0,1}
                for lw, coeff in strassen_W[l].items():
                    k_local = lw // 2
                    i_local = lw % 2
                    k_global = k0 + k_local
                    W[k_global*3 + i_local, col] += coeff
                
                col += 1
        
        # Block 2: i in {0,1}, j in {2}, k in {0,...,7}
        # Schoolbook: 2 * 1 * 8 = 16 terms
        for i in range(2):
            for k in range(8):
                U[i*3 + 2, col] = 1.0
                V[2*8 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        
        # Block 3: i in {2}, j in {0,1,2}, k in {0,...,7}
        # Schoolbook: 1 * 3 * 8 = 24 terms
        for j in range(3):
            for k in range(8):
                U[2*3 + j, col] = 1.0
                V[j*8 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the block-diagonal decomposition
    result = build_block_diagonal()
    
    if validate(result["U"], result["V"], result["W"]):
        return result
    else:
        return schoolbook()

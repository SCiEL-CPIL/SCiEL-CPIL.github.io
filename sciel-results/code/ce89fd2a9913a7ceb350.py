import numpy as np
import time

n, m, p = 2, 5, 6
TOTAL_TERMS = n * m * p  # 60

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def check_decomposition(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def compute_tensor_residuals(U, V, W):
        R = U.shape[1]
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    res[idx] = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    idx += 1
        return res - 1.0

    def strassen_2x2_block(rank2x2, i_offset, j_offset, k_offset):
        """
        Construct Strassen-style 7-term decomposition for a 2x2x2 block
        embedded in the 2x5x6 problem.
        
        For the sub-block: C[i_offset:i_offset+2, k_offset:k_offset+2] = 
                           A[i_offset:i_offset+2, j_offset:j_offset+2] @ B[j_offset:j_offset+2, k_offset:k_offset+2]
        
        Strassen for 2x2x2:
        C11 = M1 - M2 + M4 - M6
        C12 = M2 + M3
        C21 = M4 + M5
        C22 = M1 - M3 + M5 - M7
        
        Where:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B21 - B22)
        M4 = A22(B12 - B11)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        """
        U_block = np.zeros((4, 7))  # 2 rows x 2 cols = 4 entries
        V_block = np.zeros((4, 7))  # 2 cols x 2 rows = 4 entries  
        W_block = np.zeros((4, 7))  # 2 rows x 2 cols = 4 entries
        
        # Index mapping for 2x2x2 block:
        # U indices: (i, j) where i in {0,1}, j in {0,1} -> idx = i*2+j
        # V indices: (j, k) where j in {0,1}, k in {0,1} -> idx = j*2+k
        # W indices: (k, i) where k in {0,1}, i in {0,1} -> idx = k*2+i
        
        # M1 = (A11+A22)(B11+B22)
        U_block[0, 0] = 1  # A11
        U_block[3, 0] = 1  # A22
        V_block[0, 0] = 1  # B11
        V_block[3, 0] = 1  # B22
        W_block[0, 0] = 1  # contributes to C11
        W_block[1, 0] = 1  # contributes to C12
        W_block[2, 0] = 1  # contributes to C21
        W_block[3, 0] = 1  # contributes to C22
        
        # M2 = (A21+A22)B11
        U_block[2, 1] = 1  # A21
        U_block[3, 1] = 1  # A22
        V_block[0, 1] = 1  # B11
        W_block[0, 1] = -1  # C11 gets -M2
        W_block[1, 1] = 1   # C12 gets +M2
        
        # M3 = A11(B21-B22)
        U_block[0, 2] = 1  # A11
        V_block[2, 2] = 1  # B21
        V_block[3, 2] = -1 # B22
        W_block[1, 2] = 1  # C12 gets +M3
        W_block[3, 2] = -1 # C22 gets -M3
        
        # M4 = A22(B12-B11)
        U_block[3, 3] = 1  # A22
        V_block[1, 3] = 1  # B12
        V_block[0, 3] = -1 # B11
        W_block[0, 3] = 1  # C11 gets +M4
        W_block[2, 3] = 1  # C21 gets +M4
        
        # M5 = (A11+A12)B22
        U_block[0, 4] = 1  # A11
        U_block[1, 4] = 1  # A12
        V_block[3, 4] = 1  # B22
        W_block[2, 4] = 1  # C21 gets +M5
        W_block[3, 4] = 1  # C22 gets +M5
        
        # M6 = (A21-A11)(B11+B12)
        U_block[2, 5] = 1  # A21
        U_block[0, 5] = -1 # A11
        V_block[0, 5] = 1  # B11
        V_block[1, 5] = 1  # B12
        W_block[0, 5] = -1 # C11 gets -M6
        
        # M7 = (A12-A22)(B21+B22)
        U_block[1, 6] = 1  # A12
        U_block[3, 6] = -1 # A22
        V_block[2, 6] = 1  # B21
        V_block[3, 6] = 1  # B22
        W_block[3, 6] = -1 # C22 gets -M7
        
        # Now embed into full matrices
        U_full = np.zeros((n * m, 7))
        V_full = np.zeros((m * p, 7))
        W_full = np.zeros((p * n, 7))
        
        for a in range(2):
            for b in range(2):
                u_idx = (i_offset + a) * m + (j_offset + b)
                U_full[u_idx, :] = U_block[a * 2 + b, :]
        
        for b in range(2):
            for c in range(2):
                v_idx = (j_offset + b) * p + (k_offset + c)
                V_full[v_idx, :] = V_block[b * 2 + c, :]
        
        for c in range(2):
            for a in range(2):
                w_idx = (k_offset + c) * n + (i_offset + a)
                W_full[w_idx, :] = W_block[c * 2 + a, :]
        
        return U_full, V_full, W_full

    def build_strassen_decomposition():
        """
        Build a decomposition using Strassen-style 2x2x2 blocks.
        For 2x5x6:
        - Split rows: 2 rows (can't split further usefully)
        - Split columns of A / rows of B: 5 = 2 + 2 + 1
        - Split cols of B / rows of C: 6 = 2 + 2 + 2
        
        Strategy: Use Strassen for 2x2x2 blocks where possible,
        and schoolbook for the remaining 1-column terms.
        """
        U_list = []
        V_list = []
        W_list = []
        
        # Partition: j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
        # Block structure:
        # - 2x2x2 blocks: (j0,j1) x (k0,k1), (j0,j1) x (k2,k3), (j0,j1) x (k4,k5)
        #                 (j2,j3) x (k0,k1), (j2,j3) x (k2,k3), (j2,j3) x (k4,k5)
        # - 2x1x2 blocks: (j4) x (k0,k1), (j4) x (k2,k3), (j4) x (k4,k5)
        
        # 2x2x2 Strassen blocks (7 terms each)
        j_blocks_2 = [(0, 1), (2, 3)]
        k_blocks_2 = [(0, 1), (2, 3), (4, 5)]
        
        for (j0, j1) in j_blocks_2:
            for (k0, k1) in k_blocks_2:
                U_b, V_b, W_b = strassen_2x2_block(7, 0, j0, k0)
                U_list.append(U_b)
                V_list.append(V_b)
                W_list.append(W_b)
        
        # 2x1x2 blocks for j=4 (schoolbook: 2*1*2 = 4 terms each)
        for (k0, k1) in k_blocks_2:
            for j in [4]:
                for i in range(2):
                    for k in [k0, k1]:
                        U_t = np.zeros((n * m, 1))
                        V_t = np.zeros((m * p, 1))
                        W_t = np.zeros((p * n, 1))
                        U_t[i * m + j, 0] = 1
                        V_t[j * p + k, 0] = 1
                        W_t[k * n + i, 0] = 1
                        U_list.append(U_t)
                        V_list.append(V_t)
                        W_list.append(W_t)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        rank = U.shape[1]
        res = compute_tensor_residuals(U, V, W)
        max_err = np.max(np.abs(res))
        
        if max_err < 1e-9:
            return U, V, W, rank
        else:
            return None, None, None, None

    # Try Strassen-based construction
    U_str, V_str, W_str, rank_str = build_strassen_decomposition()
    if U_str is not None:
        return {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}

    # Fallback: schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

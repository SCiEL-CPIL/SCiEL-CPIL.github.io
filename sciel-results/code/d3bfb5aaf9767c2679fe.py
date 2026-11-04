import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for M(n, m, p)
    def schoolbook(n, m, p):
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

    # Helper to create Strassen decomposition for M(2, 2, 2)
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # u, v, w vectors for the 7 terms
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U[i*2+j, r] = u_r[i] * v_r[j]
        # V[j*2+k, r] = v_r[j] * w_r[k]
        # W[k*2+i, r] = w_r[k] * u_r[i]
        
        terms = [
            ([1, 1], [1, 1], [1, 1]),   # M1
            ([0, 1], [1, 0], [1, 0]),   # M2
            ([1, 0], [1, -1], [0, 1]),  # M3
            ([0, 1], [-1, 1], [1, 1]),  # M4
            ([1, 1], [0, 1], [-1, 1]),  # M5
            ([-1, 1], [1, 1], [0, 1]),  # M6
            ([0, 1], [1, -1], [1, 0])   # M7
        ]
        
        for r, (u, v, w) in enumerate(terms):
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j, r] = u[i] * v[j]
            for j in range(2):
                for k in range(2):
                    V[j * 2 + k, r] = v[j] * w[k]
            for k in range(2):
                for i in range(2):
                    W[k * 2 + i, r] = w[k] * u[i]
                    
        return U, V, W

    # Define blocks
    # J partitions: 0-1, 2-3, 4
    # K partitions: 0-1, 2-3, 4
    
    blocks = []
    
    # Block 1: J 0-1, K 0-1 (2x2x2) -> Strassen
    blocks.append(('S', 0, 0, 2, 2))
    # Block 2: J 0-1, K 2-3 (2x2x2) -> Strassen
    blocks.append(('S', 0, 2, 2, 2))
    # Block 3: J 0-1, K 4 (2x2x1) -> Schoolbook
    blocks.append(('B', 0, 4, 2, 1))
    
    # Block 4: J 2-3, K 0-1 (2x2x2) -> Strassen
    blocks.append(('S', 2, 0, 2, 2))
    # Block 5: J 2-3, K 2-3 (2x2x2) -> Strassen
    blocks.append(('S', 2, 2, 2, 2))
    # Block 6: J 2-3, K 4 (2x2x1) -> Schoolbook
    blocks.append(('B', 2, 4, 2, 1))
    
    # Block 7: J 4, K 0-1 (2x1x2) -> Schoolbook
    blocks.append(('B', 4, 0, 1, 2))
    # Block 8: J 4, K 2-3 (2x1x2) -> Schoolbook
    blocks.append(('B', 4, 2, 1, 2))
    # Block 9: J 4, K 4 (2x1x1) -> Schoolbook
    blocks.append(('B', 4, 4, 1, 1))
    
    # Calculate total rank
    total_rank = 0
    for b in blocks:
        if b[0] == 'S':
            total_rank += 7
        else:
            total_rank += 2 * b[3] * b[4]
            
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    for b in blocks:
        typ, j_start, k_start, j_size, k_size = b
        
        if typ == 'S':
            U_b, V_b, W_b = strassen_222()
        else:
            U_b, V_b, W_b = schoolbook(2, j_size, k_size)
            
        for r in range(U_b.shape[1]):
            # Map U
            for i in range(2):
                for j in range(j_size):
                    U[i * m + (j_start + j), col_offset + r] = U_b[i * j_size + j, r]
            # Map V
            for j in range(j_size):
                for k in range(k_size):
                    V[(j_start + j) * p + (k_start + k), col_offset + r] = V_b[j * k_size + k, r]
            # Map W
            for k in range(k_size):
                for i in range(2):
                    W[(k_start + k) * n + i, col_offset + r] = W_b[k * 2 + i, r]
                    
        col_offset += U_b.shape[1]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 2, 5, 5
    
    # Check rank
    print(f"Rank: {U.shape[1]}")
    
    # Check correctness with random matrices
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    
    # Standard product
    C_expected = A @ B
    
    # Tensor product
    # C_ik = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ? No.
    # The decomposition is T_ijk = sum_r U_... V_... W_...
    # C_ik = sum_j A_ij B_jk = sum_ijk T_ijk A_ij B_jk
    # T_ijk = sum_r u_r[i] v_r[j] w_r[k]
    # But U, V, W are flattened.
    # U[i*m+j, r] = u_r[i] v_r[j]
    # V[j*p+k, r] = v_r[j] w_r[k]
    # W[k*n+i, r] = w_r[k] u_r[i]
    
    # We can reconstruct T_ijk by:
    # T_ijk = sum_r (U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]) / (u_r[i] v_r[j] w_r[k] * u_r[i] v_r[j] w_r[k] * w_r[k] u_r[i])?
    # This is messy.
    
    # Easier check:
    # C_ik = sum_r (sum_j A_ij U[i*m+j, r]) * (sum_k B_jk V[j*p+k, r]) * W[k*n+i, r] ?
    # No.
    # M_r = (sum_ij U[i*m+j, r] A_ij) * (sum_jk V[j*p+k, r] B_jk)
    # C_ik += W[k*n+i, r] * M_r
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        # M_r = (A dot U_col) dot (B dot V_col)?
        # U_col is vector of length n*m. Reshape to n x m.
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        
        # L_r(A) = sum_ij U_r[i,j] A_ij = trace(U_r^T A)
        # L_r(B) = sum_jk V_r[j,k] B_jk = trace(V_r^T B)
        M_r = np.sum(U_r * A) * np.sum(V_r * B)
        
        # C_ik += W_r[k,i] * M_r
        C_computed += M_r * W_r.T # W_r is p x n, W_r[k,i] is W_r.T[i,k]
        
    # Check if C_computed == C_expected
    diff = np.max(np.abs(C_computed - C_expected))
    print(f"Max diff: {diff}")
    if diff < 1e-9:
        print("Decomposition is correct.")
    else:
        print("Decomposition is INCORRECT.")

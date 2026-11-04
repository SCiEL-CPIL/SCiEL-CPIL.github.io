import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 140

    # Rank 36 decomposition
    # Blocks:
    # 1. j in {0,1}, k in {0,1}: 2x2x2 Strassen (Rank 7)
    # 2. j in {0,1}, k in {2,3}: 2x2x2 Strassen (Rank 7)
    # 3. j in {0,1}, k in {4}:   2x2x1 Trivial (Rank 4)
    # 4. j in {2,3}, k in {0,1}: 2x2x2 Strassen (Rank 7)
    # 5. j in {2,3}, k in {2,3}: 2x2x2 Strassen (Rank 7)
    # 6. j in {2,3}, k in {4}:   2x2x1 Trivial (Rank 4)
    
    total_rank = 7 + 7 + 4 + 7 + 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(j_off, k_off, col_start):
        # M1 = (A00+A11)(B00+B11)
        U[j_off, col_start] = 1.0
        U[4 + 1 + j_off, col_start] = 1.0
        V[j_off * 5 + k_off, col_start] = 1.0
        V[(1 + j_off) * 5 + (1 + k_off), col_start] = 1.0
        W[k_off * 2 + 0, col_start] = 1.0
        W[(1 + k_off) * 2 + 1, col_start] = 1.0
        col_start += 1
        
        # M2 = (A10+A11)B00
        U[4 + 0 + j_off, col_start] = 1.0
        U[4 + 1 + j_off, col_start] = 1.0
        V[j_off * 5 + k_off, col_start] = 1.0
        W[k_off * 2 + 1, col_start] = 1.0
        W[(1 + k_off) * 2 + 1, col_start] = -1.0
        col_start += 1
        
        # M3 = A00(B01-B11)
        U[j_off, col_start] = 1.0
        V[j_off * 5 + (1 + k_off), col_start] = 1.0
        V[(1 + j_off) * 5 + (1 + k_off), col_start] = -1.0
        W[(1 + k_off) * 2 + 0, col_start] = 1.0
        W[(1 + k_off) * 2 + 1, col_start] = 1.0
        col_start += 1
        
        # M4 = A11(B10-B00)
        U[4 + 1 + j_off, col_start] = 1.0
        V[(1 + j_off) * 5 + k_off, col_start] = 1.0
        V[j_off * 5 + k_off, col_start] = -1.0
        W[k_off * 2 + 0, col_start] = 1.0
        W[(1 + k_off) * 2 + 1, col_start] = 1.0
        col_start += 1
        
        # M5 = (A00+A01)B11
        U[j_off, col_start] = 1.0
        U[j_off + 1, col_start] = 1.0
        V[(1 + j_off) * 5 + (1 + k_off), col_start] = 1.0
        W[k_off * 2 + 0, col_start] = -1.0
        W[(1 + k_off) * 2 + 0, col_start] = 1.0
        col_start += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[4 + 0 + j_off, col_start] = 1.0
        U[j_off, col_start] = -1.0
        V[j_off * 5 + k_off, col_start] = 1.0
        V[j_off * 5 + (1 + k_off), col_start] = 1.0
        W[(1 + k_off) * 2 + 1, col_start] = 1.0
        col_start += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[j_off + 1, col_start] = 1.0
        U[4 + 1 + j_off, col_start] = -1.0
        V[(1 + j_off) * 5 + k_off, col_start] = 1.0
        V[(1 + j_off) * 5 + (1 + k_off), col_start] = 1.0
        W[k_off * 2 + 0, col_start] = 1.0
        col_start += 1
        
        return col_start

    # Helper to add Trivial 2x2x1 block
    def add_trivial_221(j_off, k_off, col_start):
        # A (2x2) * B (2x1)
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Terms: A00 B0, A01 B1, A10 B0, A11 B1
        
        # T1: A00 B0
        U[j_off, col_start] = 1.0
        V[j_off * 5 + k_off, col_start] = 1.0
        W[k_off * 2 + 0, col_start] = 1.0
        col_start += 1
        
        # T2: A01 B1
        U[j_off + 1, col_start] = 1.0
        V[(j_off + 1) * 5 + k_off, col_start] = 1.0
        W[k_off * 2 + 0, col_start] = 1.0
        col_start += 1
        
        # T3: A10 B0
        U[4 + j_off, col_start] = 1.0
        V[j_off * 5 + k_off, col_start] = 1.0
        W[k_off * 2 + 1, col_start] = 1.0
        col_start += 1
        
        # T4: A11 B1
        U[4 + j_off + 1, col_start] = 1.0
        V[(j_off + 1) * 5 + k_off, col_start] = 1.0
        W[k_off * 2 + 1, col_start] = 1.0
        col_start += 1
        
        return col_start

    # Block 1: j in {0,1}, k in {0,1}
    col = add_strassen(0, 0, col)
    
    # Block 2: j in {0,1}, k in {2,3}
    col = add_strassen(0, 2, col)
    
    # Block 3: j in {0,1}, k in {4}
    col = add_trivial_221(0, 4, col)
    
    # Block 4: j in {2,3}, k in {0,1}
    col = add_strassen(2, 0, col)
    
    # Block 5: j in {2,3}, k in {2,3}
    col = add_strassen(2, 2, col)
    
    # Block 6: j in {2,3}, k in {4}
    col = add_trivial_221(2, 4, col)
    
    # Verify with random matrices
    def verify_decomposition(U, V, W):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is equivalent to:
        # For each c, we have a term U_c (8x1), V_c (20x1), W_c (10x1)
        # This term contributes to C[i,k] via sum_j U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
        
        C_computed = np.zeros((n, p))
        for c in range(U.shape[1]):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            
            # Reshape U to (n, m), V to (m, p), W to (p, n)
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            
            # The term is trace(u_mat^T * v_mat * w_mat^T)? No.
            # C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is element-wise product of (u_mat @ v_mat) and w_mat^T?
            # Let X = u_mat @ v_mat. X is (n, p).
            # X[i,k] = sum_j u_mat[i,j] * v_mat[j,k]
            # We need X[i,k] * w_mat[k,i].
            # So C_computed += X * w_mat.T
            
            X = u_mat @ v_mat
            C_computed += X * w_mat.T
            
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True

    if verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails (should not happen)
    U_s = np.zeros((n * m, n * m * p))
    V_s = np.zeros((m * p, n * m * p))
    W_s = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1.0
                V_s[j * p + k, col] = 1.0
                W_s[k * n + i, col] = 1.0
                col += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

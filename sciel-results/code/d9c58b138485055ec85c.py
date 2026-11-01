import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = 2, 2, 10
    # A is n×m = 2×2, B is m×p = 2×10, C is n×p = 2×10
    # U shape (n*m, rank) = (4, rank)
    # V shape (m*p, rank) = (20, rank)
    # W shape (p*n, rank) = (20, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Strassen's 7-term decomposition for 2×2×2
    # For matrices A (2×2), B (2×2), C (2×2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen terms as (u_vec, v_vec, w_vec) where:
    # u_vec = [A11, A12, A21, A22] (row-major 2×2)
    # v_vec = [B11, B12, B21, B22] (row-major 2×2)  
    # w_vec = [C11, C12, C21, C22] (row-major 2×2)
    
    strassen_terms = [
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A21+A22)B11 -> C21 += 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3: A11(B21-B22) -> C12 += 1, C22 += 1
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4: A22(B12-B21) -> C11 += 1, C21 += 1
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A12-A22)(B21+B22) -> C11 += 1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Build rank-35 decomposition: 5 blocks × 7 terms each
    R = 35
    U = np.zeros((n * m, R), dtype=float)  # (4, 35)
    V = np.zeros((m * p, R), dtype=float)  # (20, 35)
    W = np.zeros((p * n, R), dtype=float)  # (20, 35)
    
    col = 0
    for q in range(5):  # 5 column pairs of B
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: indices 0..3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
            U[0, col] = a_vec[0]  # A[0,0]
            U[1, col] = a_vec[1]  # A[0,1]
            U[2, col] = a_vec[2]  # A[1,0]
            U[3, col] = a_vec[3]  # A[1,1]
            
            # V: indices 0..19, for block q (columns 2q, 2q+1 of B)
            # B[0, 2q] is at index 0*10 + 2q = 2q
            # B[0, 2q+1] is at index 0*10 + 2q+1 = 2q+1
            # B[1, 2q] is at index 1*10 + 2q = 10+2q
            # B[1, 2q+1] is at index 1*10 + 2q+1 = 10+2q+1
            V[2*q, col] = b_vec[0]       # B[0, 2q]
            V[2*q+1, col] = b_vec[1]     # B[0, 2q+1]
            V[10+2*q, col] = b_vec[2]    # B[1, 2q]
            V[10+2*q+1, col] = b_vec[3]  # B[1, 2q+1]
            
            # W: indices 0..19, for block q (columns 2q, 2q+1 of C)
            # C[0, 2q] is at index (2q)*2 + 0 = 4q
            # C[0, 2q+1] is at index (2q+1)*2 + 0 = 4q+2
            # C[1, 2q] is at index (2q)*2 + 1 = 4q+1
            # C[1, 2q+1] is at index (2q+1)*2 + 1 = 4q+3
            W[4*q, col] = c_vec[0]       # C[0, 2q]
            W[4*q+2, col] = c_vec[1]     # C[0, 2q+1]
            W[4*q+1, col] = c_vec[2]     # C[1, 2q]
            W[4*q+3, col] = c_vec[3]     # C[1, 2q+1]
            
            col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n=2, m=2, p=10, n_trials=20):
        R = U.shape[1]
        # Build tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        T = np.zeros((n, m, p))
        for r in range(R):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        
        np.random.seed(12345)
        for _ in range(n_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.einsum('ijk,ij,jk->ik', T, A, B)
            if not np.allclose(C_computed, C_expected, atol=1e-10, rtol=1e-10):
                return False
        return True
    
    is_valid = verify(U, V, W)
    
    if not is_valid:
        # Fallback: naive rank-40 decomposition
        R_naive = n * m * p  # 40
        U_naive = np.zeros((n * m, R_naive), dtype=float)
        V_naive = np.zeros((m * p, R_naive), dtype=float)
        W_naive = np.zeros((p * n, R_naive), dtype=float)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j, idx] = 1.0
                    V_naive[j * p + k, idx] = 1.0
                    W_naive[k * n + i, idx] = 1.0
                    idx += 1
        
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

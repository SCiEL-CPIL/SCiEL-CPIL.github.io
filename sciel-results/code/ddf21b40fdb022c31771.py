import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS
    # 3x3 blocks of 2x2 matrices
    block_size = 2
    num_blocks = 3  # 6 / 2 = 3
    strassen_rank = 7
    total_rank = num_blocks * num_blocks * num_blocks * strassen_rank  # 9*3*7 = 189
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                # Strassen's 7 terms for 2x2 multiplication
                # M = A_block[i,k], N = B_block[k,j]
                # M elements: m00, m01, m10, m11
                # N elements: n00, n01, n10, n11
                
                # U indices for A_block[i,k]
                m00_idx = (i * block_size) * m + k * block_size
                m01_idx = (i * block_size) * m + k * block_size + 1
                m10_idx = (i * block_size + 1) * m + k * block_size
                m11_idx = (i * block_size + 1) * m + k * block_size + 1
                
                # V indices for B_block[k,j]
                n00_idx = (k * block_size) * p + j * block_size
                n01_idx = (k * block_size) * p + j * block_size + 1
                n10_idx = (k * block_size + 1) * p + j * block_size
                n11_idx = (k * block_size + 1) * p + j * block_size + 1
                
                # W indices for C_block[i,j]
                c00_idx = (j * block_size) * n + i * block_size
                c01_idx = (j * block_size + 1) * n + i * block_size
                c10_idx = (j * block_size) * n + i * block_size + 1
                c11_idx = (j * block_size + 1) * n + i * block_size + 1
                
                # Strassen terms:
                # p1 = (m00+m11)(n00+n11)
                # p2 = (m10+m11)n00
                # p3 = m00(n01-n11)
                # p4 = m11(n10-n00)
                # p5 = (m00+m01)n11
                # p6 = (m10-m00)(n00+n01)
                # p7 = (m01-m11)(n10+n11)
                
                terms = [
                    # (U_coeffs, V_coeffs, W_coeffs)
                    # p1: (m00+m11)(n00+n11)
                    ([(m00_idx, 1), (m11_idx, 1)], 
                     [(n00_idx, 1), (n11_idx, 1)],
                     [(c00_idx, 1), (c11_idx, 1)]),
                    # p2: (m10+m11)n00
                    ([(m10_idx, 1), (m11_idx, 1)], 
                     [(n00_idx, 1)],
                     [(c10_idx, 1)]),
                    # p3: m00(n01-n11)
                    ([(m00_idx, 1)], 
                     [(n01_idx, 1), (n11_idx, -1)],
                     [(c01_idx, 1), (c11_idx, 1)]),
                    # p4: m11(n10-n00)
                    ([(m11_idx, 1)], 
                     [(n10_idx, 1), (n00_idx, -1)],
                     [(c00_idx, 1), (c10_idx, 1)]),
                    # p5: (m00+m01)n11
                    ([(m00_idx, 1), (m01_idx, 1)], 
                     [(n11_idx, 1)],
                     [(c00_idx, -1), (c01_idx, 1)]),
                    # p6: (m10-m00)(n00+n01)
                    ([(m10_idx, 1), (m00_idx, -1)], 
                     [(n00_idx, 1), (n01_idx, 1)],
                     [(c11_idx, 1)]),
                    # p7: (m01-m11)(n10+n11)
                    ([(m01_idx, 1), (m11_idx, -1)], 
                     [(n10_idx, 1), (n11_idx, 1)],
                     [(c00_idx, 1)]),
                ]
                
                for u_coeffs, v_coeffs, w_coeffs in terms:
                    for idx, coeff in u_coeffs:
                        U[idx][col] += coeff
                    for idx, coeff in v_coeffs:
                        V[idx][col] += coeff
                    for idx, coeff in w_coeffs:
                        W[idx][col] += coeff
                    col += 1
    
    # Verify correctness
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for random A, B, verify W^T @ (V^T @ B) == A @ B (reshaped)
        # Actually: C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is: C = sum_r (U_col_r outer V_col_r) @ ... no
        # The tensor is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a indexes A, b indexes B, c indexes C
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # So T should satisfy: sum_k A[i,k]*B[k,j] = sum_r U[i*m+k, r]*V[k*p+j, r]*W[j*n+i, r]
        
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                val = 0.0
                for k in range(m):
                    a_idx = i * m + k
                    b_idx = k * p + j
                    c_idx = j * n + i
                    for r in range(total_rank):
                        val += U_np[a_idx, r] * V_np[b_idx, r] * W_np[c_idx, r]
                C_computed[i, j] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}

import time
import numpy as np

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140
    
    def reduce_rank(U, V, W, max_iter=100, tol=1e-8):
        """Try to reduce rank by finding linear dependencies."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Try to find a column that's approximately linearly dependent
            # Use SVD on U to find near-dependencies
            try:
                U_svd = np.linalg.svd(U, full_matrices=False)
                singular_values = U_svd[1]
                # Find smallest singular value
                min_sv = np.min(singular_values)
                if min_sv < tol:
                    # Find which column is dependent
                    idx = np.argmin(singular_values)
                    # Remove this column and check if decomposition still works
                    mask = np.ones(rank, dtype=bool)
                    mask[idx] = False
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    # Verify
                    if _verify_fast(U_new, V_new, W_new, n, m, p):
                        U, V, W = U_new, V_new, W_new
                        rank = U.shape[1]
                        continue
            except:
                break
            break
        return U, V, W
    
    def _verify_fast(U, V, W, n, m, p, tol=1e-8):
        """Fast verification using tensor contraction."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Define coefficients for 4x4x4 Strassen decomposition
    # Macro coefficients
    A_macro = [
        [[1, 0], [0, 1]],  # t=0: A00+A11
        [[0, 0], [1, 1]],  # t=1: A10+A11
        [[1, 0], [0, 0]],  # t=2: A00
        [[0, 0], [0, 1]],  # t=3: A11
        [[1, 1], [0, 0]],  # t=4: A00+A01
        [[-1, 0], [1, 0]], # t=5: A10-A00
        [[0, 1], [0, -1]]  # t=6: A01-A11
    ]
    
    B_macro = [
        [[1, 0], [0, 1]],  # t=0: B00+B11
        [[1, 0], [0, 0]],  # t=1: B00
        [[0, 1], [0, -1]], # t=2: B01-B11
        [[-1, 0], [1, 0]], # t=3: B10-B00
        [[0, 0], [0, 1]],  # t=4: B11
        [[1, 1], [0, 0]],  # t=5: B00+B01
        [[0, 1], [1, 1]]   # t=6: B10+B11
    ]
    
    # C_macro[t][b][c] is the coefficient of Z_t in C_bc
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    C_macro = [
        [[1, 0], [0, 1]], # P0: C00(1), C11(1)
        [[0, 0], [1, -1]],# P1: C10(1), C11(-1)
        [[0, 1], [0, 1]], # P2: C01(1), C11(1)
        [[1, 0], [1, 0]], # P3: C00(1), C10(1)
        [[-1, 1], [0, 0]],# P4: C00(-1), C01(1)
        [[0, 0], [0, 1]], # P5: C11(1)
        [[1, 0], [0, 0]]  # P6: C00(1)
    ]
    
    # Micro coefficients (standard Strassen for 2x2)
    A_micro = [
        [[1, 0], [0, 1]],  # s=0
        [[0, 0], [1, 1]],  # s=1
        [[1, 0], [0, 0]],  # s=2
        [[0, 0], [0, 1]],  # s=3
        [[1, 1], [0, 0]],  # s=4
        [[-1, 0], [1, 0]], # s=5
        [[0, 1], [0, -1]]  # s=6
    ]
    
    B_micro = [
        [[1, 0], [0, 1]],  # s=0
        [[1, 0], [0, 0]],  # s=1
        [[0, 1], [0, -1]], # s=2
        [[-1, 0], [1, 0]], # s=3
        [[0, 0], [0, 1]],  # s=4
        [[1, 1], [0, 0]],  # s=5
        [[0, 1], [1, 1]]   # s=6
    ]
    
    # C_micro[s][i][j] is coefficient of mu_s in Z_ij
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    C_micro = [
        [[1, 0], [0, 1]], # M0: Z00(1), Z11(1)
        [[0, 0], [1, -1]],# M1: Z10(1), Z11(-1)
        [[0, 1], [0, 1]], # M2: Z01(1), Z11(1)
        [[1, 0], [1, 0]], # M3: Z00(1), Z10(1)
        [[-1, 1], [0, 0]],# M4: Z00(-1), Z01(1)
        [[0, 0], [0, 1]], # M5: Z11(1)
        [[1, 0], [0, 0]]  # M6: Z00(1)
    ]
    
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            # Fill U
            for b in range(2):
                for c in range(2):
                    a_macro_coeff = A_macro[t][b][c]
                    if a_macro_coeff == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            a_micro_coeff = A_micro[s][i][j]
                            if a_micro_coeff == 0:
                                continue
                            # A index: block b,c, elem i,j -> (b*2+i)*4 + (c*2+j)
                            a_idx = (b * 2 + i) * 4 + (c * 2 + j)
                            U_444[a_idx, r] += a_macro_coeff * a_micro_coeff
                            
            # Fill V
            for b in range(2):
                for c in range(2):
                    b_macro_coeff = B_macro[t][b][c]
                    if b_macro_coeff == 0:
                        continue
                    for k in range(2):
                        for l in range(2):
                            b_micro_coeff = B_micro[s][k][l]
                            if b_micro_coeff == 0:
                                continue
                            # B index: block b,c, elem k,l -> (b*2+k)*4 + (c*2+l)
                            v_idx = (b * 2 + k) * 4 + (c * 2 + l)
                            V_444[v_idx, r] += b_macro_coeff * b_micro_coeff
                            
            # Fill W
            for b in range(2):
                for c in range(2):
                    c_macro_coeff = C_macro[t][b][c]
                    if c_macro_coeff == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            c_micro_coeff = C_micro[s][i][j]
                            if c_micro_coeff == 0:
                                continue
                            # C index: block b,c, elem i,j -> row b*2+i, col c*2+j
                            # W index: col * 4 + row
                            w_row = b * 2 + i
                            w_col = c * 2 + j
                            w_idx = w_col * 4 + w_row
                            W_444[w_idx, r] += c_macro_coeff * c_micro_coeff
                            
    # Construct 4x4x8
    rank_full = 98
    U_full = np.zeros((16, rank_full))
    V_full = np.zeros((32, rank_full))
    W_full = np.zeros((32, rank_full))
    
    U_full[:, 0:49] = U_444
    V_full[0:16, 0:49] = V_444
    W_full[0:16, 0:49] = W_444
    
    U_full[:, 49:98] = U_444
    V_full[16:32, 49:98] = V_444
    W_full[16:32, 49:98] = W_444
    
    # Try to reduce rank
    U_reduced, V_reduced, W_reduced = reduce_rank(U_full, V_full, W_full)
    
    U_list = U_reduced.tolist()
    V_list = V_reduced.tolist()
    W_list = W_reduced.tolist()
    
    # Verify
    if _verify_fast(np.array(U_list, dtype=float), np.array(V_list, dtype=float), np.array(W_list, dtype=float), n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to original
        U_list_orig = U_full.tolist()
        V_list_orig = V_full.tolist()
        W_list_orig = W_full.tolist()
        if _verify_fast(U_full, V_full, W_full, n, m, p):
            return {"U": U_list_orig, "V": V_list_orig, "W": W_list_orig}
        else:
            # Fallback
            rank_school = n * m * p
            U = [[0] * rank_school for _ in range(n * m)]
            V = [[0] * rank_school for _ in range(m * p)]
            W = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}

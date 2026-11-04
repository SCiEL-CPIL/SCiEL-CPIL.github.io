"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 145.0

    terms = []  # list of (a_dict, b_dict, c_dict)

    # Helper: add a 2x2x2 Strassen block at given offsets
    def add_strassen_2x2x2(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        # M1
        terms.append((
            {(i0, j0): 1, (i1, j1): 1},
            {(j0, k0): 1, (j1, k1): 1},
            {(i0, k0): 1, (i0, k1): 1, (i1, k0): 1, (i1, k1): 1}
        ))
        # M2
        terms.append((
            {(i0, j1): 1, (i1, j1): 1},
            {(j0, k0): 1},
            {(i0, k0): -1, (i1, k1): 1}
        ))
        # M3
        terms.append((
            {(i0, j0): 1},
            {(j1, k1): 1, (j0, k0): -1},
            {(i0, k0): 1, (i1, k0): -1}
        ))
        # M4
        terms.append((
            {(i1, j0): 1},
            {(j1, k0): 1, (j0, k0): -1},
            {(i1, k0): 1}
        ))
        # M5
        terms.append((
            {(i0, j0): 1, (i0, j1): 1},
            {(j1, k1): 1},
            {(i0, k1): -1, (i1, k1): 1}
        ))
        # M6
        terms.append((
            {(i1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0, k1): 1},
            {(i0, k0): 1}
        ))
        # M7
        terms.append((
            {(i0, j1): 1, (i1, j1): -1},
            {(j1, k0): 1, (j1, k1): 1},
            {(i0, k1): 1, (i1, k1): -1}
        ))

    # Block-level Strassen for 4x4x4
    block_strassen_4x4x4 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): -1, (1,1): 1}),
        ({(0,0): 1}, {(1,1): 1, (0,0): -1}, {(0,0): 1, (1,0): -1}),
        ({(1,0): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): -1, (1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,1): 1, (1,1): -1}),
    ]

    # 2x2x2 Strassen coefficients
    strassen_2x2x2_coeffs = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        ({(0,1):1, (1,1):1}, {(0,0):1}, {(0,0):-1, (1,1):1}),
        ({(0,0):1}, {(1,1):1, (0,0):-1}, {(0,0):1, (1,0):-1}),
        ({(1,0):1}, {(1,0):1, (0,0):-1}, {(1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):-1}),
    ]

    # Build 4x4x4 part (49 terms)
    for ca, cb, cc in block_strassen_4x4x4:
        for alpha, beta, gamma in strassen_2x2x2_coeffs:
            a_coeffs = {}
            for (a, c), val_a in alpha.items():
                for (i_g, j_g), val_ca in ca.items():
                    i = 2 * i_g + a
                    j = 2 * j_g + c
                    a_coeffs[(i, j)] = a_coeffs.get((i, j), 0) + val_a * val_ca

            b_coeffs = {}
            for (c, d), val_b in beta.items():
                for (j_g, k_g), val_cb in cb.items():
                    j = 2 * j_g + c
                    k = 2 * k_g + d
                    b_coeffs[(j, k)] = b_coeffs.get((j, k), 0) + val_b * val_cb

            c_coeffs = {}
            for (a, d), val_g in gamma.items():
                for (i_g, k_g), val_cc in cc.items():
                    i = 2 * i_g + a
                    k = 2 * k_g + d
                    c_coeffs[(i, k)] = c_coeffs.get((i, k), 0) + val_g * val_cc

            terms.append((a_coeffs, b_coeffs, c_coeffs))

    # Build 4x4x2 part (28 terms)
    for i_g in range(2):
        for j_g in range(2):
            add_strassen_2x2x2(2 * i_g, 2 * j_g, 4)

    # Build matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for col, (a_coeffs, b_coeffs, c_coeffs) in enumerate(terms):
        for (i, j), coeff in a_coeffs.items():
            U[i * m + j, col] = coeff
        for (j, k), coeff in b_coeffs.items():
            V[j * p + k, col] = coeff
        for (i, k), coeff in c_coeffs.items():
            W[k * n + i, col] = coeff

    def verify_decomp(U_mat, V_mat, W_mat):
        T_reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
        return np.allclose(T_reconstructed, 1.0, atol=1e-10)

    if not verify_decomp(U, V, W):
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Structural rank reduction via linear algebra:
    # Each rank-1 term t_r = u_r ⊗ v_r ⊗ w_r can be viewed as a vector in 
    # R^(nm * mp * pn) by flattening the tensor. If we have linear dependencies
    # among these vectors, we can reduce the rank.
    
    # However, a more practical approach: try all subsets of size rank-1, rank-2, etc.
    # and check if the remaining terms can still span the multiplication tensor.
    
    # Better approach: Use the fact that for matrix multiplication, the tensor is
    # T[i,j,k] = delta_ik (when multiplying A*B). We can try to find a basis.
    
    # Most practical: greedy removal with verification
    def try_reduce_rank_greedy(U_mat, V_mat, W_mat, max_time=100.0):
        start = time.time()
        current_U = U_mat.copy()
        current_V = V_mat.copy()
        current_W = W_mat.copy()
        current_rank = current_U.shape[1]
        
        def fast_verify(U_v, V_v, W_v):
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.dot(U_v[i*m+j, :], V_v[j*p+k, :] * W_v[k*n+i, :])
            return np.allclose(T, 1.0, atol=1e-10)
        
        removed = set()
        improved = True
        while improved:
            improved = False
            for col in range(current_rank):
                if time.time() - start > max_time:
                    break
                if col in removed:
                    continue
                
                mask = [i for i in range(current_rank) if i != col and i not in removed]
                if len(mask) == 0:
                    break
                    
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                
                if fast_verify(U_test, V_test, W_test):
                    removed.add(col)
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    current_rank = len(mask)
                    improved = True
                    break
        
        return current_U, current_V, current_W, current_rank

    U_reduced, V_reduced, W_reduced, new_rank = try_reduce_rank_greedy(U, V, W, max_time=100.0)
    
    if new_rank < rank:
        return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

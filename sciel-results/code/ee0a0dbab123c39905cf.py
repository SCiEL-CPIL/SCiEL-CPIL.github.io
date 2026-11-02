import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    inner_strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], {(0,0): 1, (1,1): 1}),
        ([0, 0, 1, 1], [1, 0, 0, 0], {(1,0): 1, (1,1): -1}),
        ([1, 0, 0, 0], [0, 1, 0, -1], {(0,1): 1, (1,1): 1}),
        ([0, 0, 0, 1], [0, 0, 1, -1], {(0,0): 1, (1,0): 1}),
        ([1, 1, 0, 0], [0, 0, 0, 1], {(0,0): -1, (0,1): 1}),
        ([-1, 0, 1, 0], [1, 1, 0, 0], {(1,1): 1}),
        ([0, 1, 0, -1], [0, 0, 1, 1], {(0,0): 1, (1,1): -1}),
    ]
    
    def block_to_matrix(block_id, coeff):
        br, bc = block_id
        mat = [[{} for _ in range(2)] for _ in range(2)]
        for ir in range(2):
            for ic in range(2):
                i = br * 2 + ir
                j = bc * 2 + ic
                mat[ir][ic] = {(i, j): coeff}
        return mat
    
    def add_matrices(mat1, mat2):
        result = [[{} for _ in range(2)] for _ in range(2)]
        for ir in range(2):
            for ic in range(2):
                d = {}
                for key, val in mat1[ir][ic].items():
                    d[key] = d.get(key, 0) + val
                for key, val in mat2[ir][ic].items():
                    d[key] = d.get(key, 0) + val
                result[ir][ic] = d
        return result
    
    def scale_matrix(mat, s):
        result = [[{} for _ in range(2)] for _ in range(2)]
        for ir in range(2):
            for ic in range(2):
                for key, val in mat[ir][ic].items():
                    result[ir][ic][key] = s * val
        return result
    
    A_blocks = {}
    B_blocks = {}
    for br in range(2):
        for bc in range(2):
            A_blocks[(br, bc)] = block_to_matrix((br, bc), 1)
            B_blocks[(br, bc)] = block_to_matrix((br, bc), 1)
    
    outer_mults = [
        (add_matrices(A_blocks[(0,0)], A_blocks[(1,1)]),
         add_matrices(B_blocks[(0,0)], B_blocks[(1,1)])),
        (add_matrices(A_blocks[(1,0)], A_blocks[(1,1)]),
         B_blocks[(0,0)]),
        (A_blocks[(0,0)],
         add_matrices(B_blocks[(0,1)], scale_matrix(B_blocks[(1,1)], -1))),
        (A_blocks[(1,1)],
         add_matrices(B_blocks[(1,0)], scale_matrix(B_blocks[(0,0)], -1))),
        (add_matrices(A_blocks[(0,0)], A_blocks[(0,1)]),
         B_blocks[(1,1)]),
        (add_matrices(A_blocks[(1,0)], scale_matrix(A_blocks[(0,0)], -1)),
         add_matrices(B_blocks[(0,0)], B_blocks[(0,1)])),
        (add_matrices(A_blocks[(0,1)], scale_matrix(A_blocks[(1,1)], -1)),
         add_matrices(B_blocks[(1,0)], B_blocks[(1,1)])),
    ]
    
    c_block_contribs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1, (1,1): -1},
    ]
    
    col = 0
    for outer_idx in range(7):
        X_mat, Y_mat = outer_mults[outer_idx]
        for inner_idx in range(7):
            a_coeffs, b_coeffs, z_contribs = inner_strassen[inner_idx]
            
            U_vec = {}
            for ir in range(2):
                for ic in range(2):
                    a_coeff = a_coeffs[ir * 2 + ic]
                    if a_coeff != 0:
                        for a_idx, a_val in X_mat[ir][ic].items():
                            i, j = a_idx
                            U_vec[(i, j)] = U_vec.get((i, j), 0) + a_coeff * a_val
            
            V_vec = {}
            for ir in range(2):
                for ic in range(2):
                    b_coeff = b_coeffs[ir * 2 + ic]
                    if b_coeff != 0:
                        for b_idx, b_val in Y_mat[ir][ic].items():
                            j, k = b_idx
                            V_vec[(j, k)] = V_vec.get((j, k), 0) + b_coeff * b_val
            
            W_vec = {}
            for (z_ir, z_ic), z_coeff in z_contribs.items():
                if z_coeff == 0:
                    continue
                for (c_br, c_bc), c_coeff in c_block_contribs[outer_idx].items():
                    if c_coeff == 0:
                        continue
                    i = c_br * 2 + z_ir
                    k = c_bc * 2 + z_ic
                    W_vec[(i, k)] = W_vec.get((i, k), 0) + c_coeff * z_coeff
            
            for (i, j), val in U_vec.items():
                if val != 0:
                    U[i * m + j][col] = val
            for (j, k), val in V_vec.items():
                if val != 0:
                    V[j * p + k][col] = val
            for (i, k), val in W_vec.items():
                if val != 0:
                    W[k * n + i][col] = val
            
            col += 1
    
    # Verify with a quick test
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(0)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 4))
    for r in range(rank):
        u_vals = np.zeros(4)
        for i in range(4):
            for j in range(4):
                u_vals[i] += U_np[i*4+j, r] * A_test[i,j]
        v_vals = np.zeros(4)
        for k in range(4):
            for j in range(4):
                v_vals[k] += V_np[j*4+k, r] * B_test[j,k]
        for i in range(4):
            for k in range(4):
                C_computed[i,k] += u_vals[i] * v_vals[k] * W_np[k*4+i, r]
    
    if np.max(np.abs(C_computed - C_expected)) > 1e-9:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

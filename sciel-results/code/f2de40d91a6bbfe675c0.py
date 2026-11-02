import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-48 decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen (rank 49) then attempts to eliminate one term
    by finding a linear dependence among the 49 rank-1 tensor terms.
    """
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    n, m, p = 4, 4, 4

    def get_U_index(row, col):
        return row * 4 + col

    def get_V_index(row, col):
        return row * 4 + col

    def get_W_index(k, i):
        return k * 4 + i  # trace convention: W[k*n+i, r]

    # Strassen 2x2 formulas:
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    block_strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for t in range(7):
        block_A, block_B, block_C = block_strassen[t]
        for s in range(7):
            scalar_A, scalar_B, scalar_C = strassen_2x2[s]
            r_idx = t * 7 + s

            for (bi, bj), block_coeff in block_A.items():
                for (sr, sc), scalar_coeff in scalar_A.items():
                    a_row = 2 * bi + sr
                    a_col = 2 * bj + sc
                    U[get_U_index(a_row, a_col), r_idx] += block_coeff * scalar_coeff

            for (bi, bj), block_coeff in block_B.items():
                for (sr, sc), scalar_coeff in scalar_B.items():
                    b_row = 2 * bi + sr
                    b_col = 2 * bj + sc
                    V[get_V_index(b_row, b_col), r_idx] += block_coeff * scalar_coeff

            for (ci, cj), block_coeff in block_C.items():
                for (sr, sc), scalar_coeff in scalar_C.items():
                    c_row = 2 * ci + sr
                    c_col = 2 * cj + sc
                    w_idx = get_W_index(c_col, c_row)
                    W[w_idx, r_idx] += block_coeff * scalar_coeff

    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by finding linear dependence
    # Flatten each term u_r ⊗ v_r ⊗ w_r to a vector of length 4096
    dim = 4096  # 16*16*16
    
    # Build the matrix of 49 flattened terms
    terms = []
    for r in range(49):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # Tensor product: T[i,j,k] = u[i]*v[j]*w[k]
        tensor = np.outer(np.outer(u, v), w).flatten()
        terms.append(tensor)
    
    terms_arr = np.array(terms, dtype=float)  # shape (49, 4096)
    
    # Check if we can eliminate one term
    # Try to find c such that sum_{r=0}^{48} c_r * terms[r] = terms[48]
    # i.e., terms[48] is in the span of terms[0:48]
    A = terms_arr[:48, :]  # 48 x 4096
    b = terms_arr[48, :]   # 4096
    
    # Solve least squares: A^T A c = A^T b
    try:
        ATA = A.T @ A  # 4096 x 4096 - too big, use QR or SVD
        # Use QR decomposition
        Q, R = np.linalg.qr(A.T, mode='economic')  # 4096 x 48
        # Q shape: 4096 x 48, R shape: 48 x 48
        # Solve R c = Q^T b
        Qtb = Q.T @ b  # 48
        c = np.linalg.solve(R, Qtb)  # 48
        
        # Verify
        reconstruction = A @ c
        residual = np.linalg.norm(b - reconstruction) / (np.linalg.norm(b) + 1e-15)
        
        if residual < 1e-6:
            # We found a dependence!
            # New coefficients: for r=0..47, multiply by (1 + c_r); drop term 48
            new_rank = 48
            U_new = np.zeros((16, new_rank), dtype=float)
            V_new = np.zeros((16, new_rank), dtype=float)
            W_new = np.zeros((16, new_rank), dtype=float)
            
            for r in range(48):
                scale = 1.0 + c[r]
                U_new[:, r] = U[:, r] * scale
                V_new[:, r] = V[:, r] * scale
                W_new[:, r] = W[:, r] * scale
            
            # Verify the decomposition
            # T_new = sum_{r=0}^{47} (1+c_r) u_r v_r w_r
            # = sum_{r=0}^{47} u_r v_r w_r + sum_{r=0}^{47} c_r u_r v_r w_r
            # = (sum_{r=0}^{48} u_r v_r w_r - u_48 v_48 w_48) + u_48 v_48 w_48
            # = sum_{r=0}^{48} u_r v_r w_r  ✓
            
            # Check that it's actually a valid decomposition by sampling
            A_test = np.random.randn(16, 16)
            B_test = np.random.randn(16, 16)
            C_direct = A_test @ B_test
            
            # Compute using the new factors
            # U_new[i*4+j, r], V_new[j*4+k, r], W_new[k*4+i, r]
            C_new = np.zeros((16, 16))
            for i in range(4):
                for j in range(4):
                    Cij = 0.0
                    for k in range(4):
                        s = 0.0
                        for r in range(48):
                            s += U_new[i*4+k, r] * V_new[k*4+j, r] * W_new[j*4+i, r]
                        Cij += s
                    C_new[i, j] = Cij
            
            err = np.linalg.norm(C_direct - C_new) / (np.linalg.norm(C_direct) + 1e-15)
            
            if err < 1e-6:
                # Convert to integer/half-integer if possible
                U_out = np.round(U_new * 2) / 2
                V_out = np.round(V_new * 2) / 2
                W_out = np.round(W_new * 2) / 2
                
                # Final verification
                C_check = np.zeros((16, 16))
                for i in range(4):
                    for j in range(4):
                        Cij = 0.0
                        for k in range(4):
                            s = 0.0
                            for r in range(48):
                                s += U_out[i*4+k, r] * V_out[k*4+j, r] * W_out[j*4+i, r]
                            Cij += s
                        C_check[i, j] = Cij
                
                err2 = np.linalg.norm(C_direct - C_check) / (np.linalg.norm(C_direct) + 1e-15)
                if err2 < 1e-6:
                    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    except Exception:
        pass

    # Fallback: return the rank-49 decomposition
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    
    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
# EVOLVE-BLOCK-END

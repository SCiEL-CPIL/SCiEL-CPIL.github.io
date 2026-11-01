"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Build the rank-21 Strassen decomposition
    rank_base = 21
    U = np.zeros((n * m, rank_base), dtype=float)
    V = np.zeros((m * p, rank_base), dtype=float)
    W = np.zeros((p * n, rank_base), dtype=float)

    # Standard Strassen 7 terms for 2x2x2 matrix multiplication
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: maps (i,j) index to coefficient
    # v_dict: maps (j,k) index to coefficient  
    # w_dict: maps (i,k) index to coefficient
    strassen_terms = [
        # Term 1: M1 = (A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # Term 2: M2 = (A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # Term 3: M3 = A11(B21-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # Term 4: M4 = A22(B12-B22)
        ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        # Term 5: M5 = (A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # Term 6: M6 = (A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # Term 7: M7 = (A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # For 2x2 x 2x6, we have 3 blocks of 2 columns each
    # Block b uses columns 2b, 2b+1 of B
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            for (i, j), val in u_coeffs.items():
                U[i * m + j][r] = val
            for (j, k_local), val in v_coeffs.items():
                k_global = 2 * b + k_local
                V[j * p + k_global][r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                W[k_global * n + i][r] = val
            r += 1

    # Build target tensor T of shape (n*m, m*p, p*n) = (4, 12, 12)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Verify the rank-21 decomposition reproduces T
    def verify_tensor(U_mat, V_mat, W_mat, tol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.allclose(T_recon, T, atol=tol)

    if not verify_tensor(U, V, W):
        # Should not happen, but return as fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Round to half-integers for cleaner output
    U_out = np.round(U * 2) / 2
    V_out = np.round(V * 2) / 2
    W_out = np.round(W * 2) / 2

    if verify_tensor(U_out, V_out, W_out):
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

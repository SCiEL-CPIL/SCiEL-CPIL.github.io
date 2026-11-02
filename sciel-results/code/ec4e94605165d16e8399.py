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

    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]

    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            for l, val in v_coeffs.items():
                if l == 0: v_idx = 2 * b
                elif l == 1: v_idx = 2 * b + 1
                elif l == 2: v_idx = 6 + 2 * b
                else: v_idx = 7 + 2 * b
                V[v_idx][r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            r += 1

    # Build the full tensor T of shape (n*m, m*p, p*n) = (4, 12, 12)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Verify the rank-21 decomposition reproduces T
    def verify_tensor(U_mat, V_mat, W_mat, T_mat, tol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.allclose(T_recon, T_mat, atol=tol)

    if not verify_tensor(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by eliminating one column
    # Check if removing any single term still allows reconstruction
    for remove_idx in range(rank_base):
        if time.time() > deadline - 5:
            break
        U_sub = np.delete(U, remove_idx, axis=1)
        V_sub = np.delete(V, remove_idx, axis=1)
        W_sub = np.delete(W, remove_idx, axis=1)

        # Build the linear system: T_flat = M @ x
        # where M[:, r] = U_sub[:,r] ⊗ V_sub[:,r] ⊗ W_sub[:,r] flattened
        M = np.zeros((n * m * m * p * p * n, rank_base - 1))
        for r in range(rank_base - 1):
            M[:, r] = np.kron(np.kron(U_sub[:, r], V_sub[:, r]), W_sub[:, r])

        T_flat = T.flatten()

        # Check if T_flat is in the column space of M
        try:
            x, residuals, rank, sv = np.linalg.lstsq(M, T_flat, rcond=None)
            T_recon = M @ x
            if np.allclose(T_recon, T_flat, atol=1e-9):
                # Found a rank-20 decomposition!
                # Absorb x into U
                U_final = U_sub * x[np.newaxis, :]
                V_final = V_sub
                W_final = W_sub

                if verify_tensor(U_final, V_final, W_final, T):
                    # Convert to list and return
                    U_out = np.round(U_final * 2) / 2
                    V_out = np.round(V_final * 2) / 2
                    W_out = np.round(W_final * 2) / 2
                    if verify_tensor(U_out, V_out, W_out, T):
                        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                    else:
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        except Exception:
            pass

    # Return the rank-21 solution
    U_out = np.round(U * 2) / 2
    V_out = np.round(V * 2) / 2
    W_out = np.round(W * 2) / 2
    if verify_tensor(U_out, V_out, W_out, T):
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

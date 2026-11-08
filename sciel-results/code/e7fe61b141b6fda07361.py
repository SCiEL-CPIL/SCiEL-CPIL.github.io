import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)

    # Rank 17 decomposition for 2x3x3 matrix multiplication
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))

    # Indices:
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j (6 entries)
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k (9 entries)
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (6 entries) [Trace convention W[k*n+i, r]]

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1

    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1

    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1

    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1

    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1

    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1

    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1

    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1

    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1

    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1

    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor"""
        rank = U_mat.shape[1]
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    actual = sum(U_mat[i*m_val+j, r] * V_mat[j*p_val+k, r] * W_mat[k*n_val+i, r] for r in range(rank))
                    if abs(actual - 1.0) > tol:
                        return False
        return True

    # Verify the initial rank-17 solution
    if not verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 17
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    # Strategy: Try to find a rank-16 decomposition
    # For each term to remove, fix two matrices' remaining columns and solve for the third
    # This is a linear system that can be solved exactly

    for term_to_remove in range(17):
        if time.time() > deadline - 5:
            break

        # Zero out the term
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, term_to_remove] = 0
        V_test[:, term_to_remove] = 0
        W_test[:, term_to_remove] = 0

        keep_indices = [r for r in range(17) if r != term_to_remove]
        found = False

        # Try fixing U and V, solve for W
        # For each (i,j,k): sum_{r in keep} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is linear in W entries
        A_sys = np.zeros((n*m*p, p*n))
        b_sys = np.zeros(n*m*p)
        for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            for r in keep_indices:
                coeff = U_test[i*m+j, r] * V_test[j*p+k, r]
                if abs(coeff) > 1e-12:
                    A_sys[idx, k*n+i] += coeff
                    b_sys[idx] = 1.0

        try:
            w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-10)
            W_try = W_test.copy()
            for r_idx, r in enumerate(keep_indices):
                for k in range(p):
                    for i in range(n):
                        W_try[k*n+i, r] = w_sol[k*n+i]
            # Note: w_sol has p*n entries total, but we need to distribute among keep_indices
            # Actually the system is: for each (i,j,k), sum over r of (U*V)_r * W_{k,i}^{(r)} = 1
            # This is NOT separable per r! We need to think differently.
        except:
            pass

        # Better approach: The system is linear in the 16*6 = 96 W variables
        # Let's set it up properly
        num_w_vars = len(keep_indices) * p * n
        A_sys2 = np.zeros((n*m*p, num_w_vars))
        b_sys2 = np.ones(n*m*p)
        
        for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            for r_idx, r in enumerate(keep_indices):
                coeff = U_test[i*m+j, r] * V_test[j*p+k, r]
                if abs(coeff) > 1e-12:
                    # Variable index: r_idx * (p*n) + k*n + i
                    A_sys2[idx, r_idx * (p*n) + k*n + i] = coeff

        try:
            w_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys2, b_sys2, rcond=1e-10)
            # Check residual
            if np.allclose(A_sys2 @ w_flat, b_sys2, atol=1e-9):
                # Build W matrix
                W_new = np.zeros((p*n, 16))
                for r_idx, r in enumerate(keep_indices):
                    for k in range(p):
                        for i in range(n):
                            W_new[k*n+i, r_idx] = w_flat[r_idx * (p*n) + k*n + i]
                
                U_new = U_test[:, keep_indices]
                V_new = V_test[:, keep_indices]
                
                if verify(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    best_rank = 16
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    found = True
                    break
        except:
            pass

        if found:
            break

        # Also try fixing U and W, solve for V
        num_v_vars = len(keep_indices) * m * p
        A_sys3 = np.zeros((n*m*p, num_v_vars))
        b_sys3 = np.ones(n*m*p)
        
        for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            for r_idx, r in enumerate(keep_indices):
                coeff = U_test[i*m+j, r] * W_test[k*n+i, r]
                if abs(coeff) > 1e-12:
                    A_sys3[idx, r_idx * (m*p) + j*p + k] = coeff

        try:
            v_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys3, b_sys3, rcond=1e-10)
            if np.allclose(A_sys3 @ v_flat, b_sys3, atol=1e-9):
                V_new = np.zeros((m*p, 16))
                for r_idx, r in enumerate(keep_indices):
                    for j in range(m):
                        for k in range(p):
                            V_new[j*p+k, r_idx] = v_flat[r_idx * (m*p) + j*p + k]
                
                U_new = U_test[:, keep_indices]
                W_new = W_test[:, keep_indices]
                
                if verify(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    best_rank = 16
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    found = True
                    break
        except:
            pass

        if found:
            break

        # Also try fixing V and W, solve for U
        num_u_vars = len(keep_indices) * n * m
        A_sys4 = np.zeros((n*m*p, num_u_vars))
        b_sys4 = np.ones(n*m*p)
        
        for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            for r_idx, r in enumerate(keep_indices):
                coeff = V_test[j*p+k, r] * W_test[k*n+i, r]
                if abs(coeff) > 1e-12:
                    A_sys4[idx, r_idx * (n*m) + i*m + j] = coeff

        try:
            u_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys4, b_sys4, rcond=1e-10)
            if np.allclose(A_sys4 @ u_flat, b_sys4, atol=1e-9):
                U_new = np.zeros((n*m, 16))
                for r_idx, r in enumerate(keep_indices):
                    for i in range(n):
                        for j in range(m):
                            U_new[i*m+j, r_idx] = u_flat[r_idx * (n*m) + i*m + j]
                
                V_new = V_test[:, keep_indices]
                W_new = W_test[:, keep_indices]
                
                if verify(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    best_rank = 16
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    found = True
                    break
        except:
            pass

    if best_rank == 16:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

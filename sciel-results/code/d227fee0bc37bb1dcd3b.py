import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def check_and_solve(U, V, r):
        """
        U: shape (n, m, r)
        V: shape (m, p, r)
        Returns W (p, n, r) if valid, else None.
        """
        W = np.zeros((p, n, r))
        b_ik = np.eye(m).flatten()
        
        # Pre-compute U_i and V_k matrices
        # U_i is m x r
        # V_k is m x r
        for i in range(n):
            for k in range(p):
                U_i = U[i, :, :]  # m x r
                V_k_mat = V[:, k, :].T  # r x m
                
                # Construct A_ik (m^2 x r)
                # A_ik[:, c] = kron(V_k_mat[c, :].T, U_i[:, c])
                # V_k_mat[c, :] is 1 x m. .T is m x 1.
                # U_i[:, c] is m x 1.
                # kron(m x 1, m x 1) is m^2 x 1.
                
                # Vectorized construction of A_ik
                # We want A_ik such that A_ik @ w = vectorized(U_i * V_k^T * w) ?
                # Actually, the equation is:
                # sum_c U_i[:, c] * V_k_mat[c, :]^T * W_ki[:, c] = I_m
                # This is: sum_c (V_k_mat[c, :]^T) (U_i[:, c])^T W_ki[c] = I_m ?
                # No, the tensor decomposition is:
                # T_{i, j, k} = sum_c U_{i, j, c} V_{j, k, c} W_{k, i, c}
                # For fixed i, k, we want sum_c U_{i, j, c} V_{j, k, c} W_{k, i, c} = delta_{j, j'}?
                # Wait, the condition is that the matrix M_{j, j'} = sum_c U_{i, j, c} V_{j', k, c} W_{k, i, c} is Identity?
                # No. The multiplication is C_{i, k} = sum_j A_{i, j} B_{j, k}.
                # The tensor T_{i, j, k} corresponds to the term A_{i, j} B_{j, k}.
                # We want sum_c U_{i, j, c} V_{j, k, c} W_{k, i, c} = 1 if j is the middle index?
                # Actually, the standard form is:
                # sum_j A_{i, j} B_{j, k} = sum_c (sum_j U_{i, j, c} A_{i, j}) (sum_j V_{j, k, c} B_{j, k}) W_{k, i, c} ?
                # No, that's not right.
                # The decomposition is:
                # A_{i, j} B_{j, k} = sum_c U_{i, j, c} V_{j, k, c} W_{k, i, c} is NOT the form.
                # The form is:
                # C_{i, k} = sum_j A_{i, j} B_{j, k} = sum_c M_c(A) N_c(B) W_{k, i, c}
                # Where M_c(A) = sum_j U_{i, j, c} A_{i, j} ? No.
                # Usually: M_c(A) = sum_{i, j} U_{i, j, c} A_{i, j} is a scalar? No.
                # The factors U, V, W are such that:
                # sum_c U_{i, j, c} V_{j, k, c} W_{k, i, c} = delta_{j, j'} is incorrect.
                
                # Let's look at the verification in the original code:
                # actual = sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c]
                # expected = 1.0 if j1 == j2 else 0.0
                # This implies that for fixed i, k, the matrix M_{j1, j2} = sum_c U_{i, j1, c} V_{j2, k, c} W_{k, i, c} must be Identity.
                # So, for fixed i, k, we have:
                # sum_c (U_{i, :, c}) (V_{:, k, c})^T W_{k, i, c} = I_m
                # Let u_c = U_{i, :, c} (vector m), v_c = V_{:, k, c} (vector m), w_c = W_{k, i, c} (scalar).
                # sum_c w_c u_c v_c^T = I_m.
                # This is a sum of rank-1 matrices.
                # We need to find w_c such that this holds.
                # Vectorizing the LHS:
                # sum_c w_c kron(v_c, u_c) = vec(I_m)
                # So A_ik[:, c] = kron(V_k_mat[c, :].T, U_i[:, c]) is correct.
                # Note: kron(v, u) where v is m, u is m.
                # np.kron(v, u) creates a matrix. If v, u are 1D, it's m^2 x 1.
                
                A_ik = np.zeros((m * m, r))
                for c in range(r):
                    # v_c is V[:, k, c] -> shape (m,)
                    # u_c is U[i, :, c] -> shape (m,)
                    # We need kron(v_c, u_c)
                    A_ik[:, c] = np.kron(V[:, k, c], U[i, :, c])
                
                try:
                    w_ik, residuals, rank_val, s = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                    if len(residuals) > 0:
                        if residuals[0] > 1e-8:
                            return None
                    else:
                        residual = np.linalg.norm(A_ik @ w_ik - b_ik)
                        if residual > 1e-8:
                            return None
                    W[k, i, :] = w_ik
                except Exception:
                    return None
        return W

    # Search for lower rank
    for r in range(rank_school - 1, 15, -1):
        if time.time() > deadline - 10:
            break
        
        # More tries for smaller ranks where success is less likely
        if r > 55:
            num_tries = 30
        elif r > 45:
            num_tries = 50
        elif r > 35:
            num_tries = 80
        else:
            num_tries = 120
        
        found = False
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.random.randn(n, m, r)
            V_rand = np.random.randn(m, p, r)
            
            W_rand = check_and_solve(U_rand, V_rand, r)
            
            if W_rand is not None:
                U_out = U_rand.reshape(n * m, r)
                V_out = V_rand.reshape(m * p, r)
                W_out = W_rand.T.reshape(p * n, r)
                
                # Final verification
                valid = True
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                expected = 1.0 if j1 == j2 else 0.0
                                actual = 0.0
                                for c in range(r):
                                    actual += U_out[i * m + j1, c] * V_out[j2 * p + k, c] * W_out[k * n + i, c]
                                if abs(actual - expected) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = r
                    best_U = U_out.copy()
                    best_V = V_out.copy()
                    best_W = W_out.copy()
                    found = True
                    break
        
        if found:
            # Continue searching for even lower rank
            continue
        else:
            # If we couldn't find a decomposition at this rank, try a few more at the next lower rank
            # but stop if time is running out
            if time.time() > deadline - 15:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

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

    # b_ik is vec(I_m)
    # We need to solve A_ik w = b_ik
    # A_ik = V_k^T \otimes U_i
    # U_i is m x r, V_k is r x m
    # U_i[:, c] V_k[c, :] is m x m
    
    def check_and_solve(U, V, r):
        """
        U: shape (n, m, r)
        V: shape (m, p, r)
        Returns W (p, n, r) if valid, else None.
        """
        W = np.zeros((p, n, r))
        for i in range(n):
            for k in range(p):
                # Extract U_i (m x r) and V_k (r x m)
                U_i = U[i, :, :] # m x r
                V_k = V[:, k, :] # m x r, so V_k^T is r x m? 
                # V[j, k, c]. We need V_k[c, j] for the outer product U_i[:, c] V_k[c, j].
                # So V_k_mat = V[:, k, :].T -> r x m
                V_k_mat = V[:, k, :].T # r x m
                
                # Construct A_ik (m^2 x r)
                # A_ik[(j1, j2), c] = U_i[j1, c] * V_k_mat[c, j2]
                # This is (V_k_mat.T \otimes U_i) ?
                # Let's verify:
                # vec(U_i[:, c] V_k_mat[c, :]) = V_k_mat[c, :]^T \otimes U_i[:, c]
                # So A_ik = sum_c (V_k_mat[c, :]^T \otimes U_i[:, c]) * W_c?
                # Actually, A_ik is the matrix such that A_ik w = vec(sum_c w_c U_i[:, c] V_k_mat[c, :])
                # The c-th column of A_ik is vec(U_i[:, c] V_k_mat[c, :]) = np.kron(V_k_mat[c, :].T, U_i[:, c])
                
                A_ik = np.zeros((m * m, r))
                for c in range(r):
                    A_ik[:, c] = np.kron(V_k_mat[c, :].T, U_i[:, c])
                
                b_ik = np.eye(m).flatten()
                
                # Check if b_ik is in the column space of A_ik
                # Use least squares and check residual
                try:
                    w_ik, residuals, rank, s = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                    if len(residuals) > 0:
                        if residuals[0] > 1e-9:
                            return None
                    else:
                        # If rank < m*m, check if residual is small
                        # Actually lstsq returns residuals only if rank == min(M, N)
                        # If rank < m*m, we need to check ||A w - b||
                        residual = np.linalg.norm(A_ik @ w_ik - b_ik)
                        if residual > 1e-9:
                            return None
                            
                    W[k, i, :] = w_ik
                except Exception:
                    return None
        return W

    # Search for lower rank
    # We try ranks from 62 down to 20 (arbitrary stop, or until time runs out)
    # We use multiple random restarts for each rank
    for r in range(rank_school - 1, 10, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        # For smaller r, we might need more tries, but let's keep it balanced
        num_tries = 10 if r > 50 else 20
        
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            # Random U and V
            # Scale matters for numerical stability, but lstsq handles it
            U_rand = np.random.randn(n, m, r)
            V_rand = np.random.randn(m, p, r)
            
            W_rand = check_and_solve(U_rand, V_rand, r)
            
            if W_rand is not None:
                # Reshape to required output format
                U_out = U_rand.reshape(n * m, r)
                V_out = V_rand.reshape(m * p, r)
                W_out = W_rand.T.reshape(p * n, r) # W[k, i, c] -> index k*n+i
                
                # Final verification to be absolutely sure
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
                            if not valid: break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_rank = r
                    best_U = U_out.copy()
                    best_V = V_out.copy()
                    best_W = W_out.copy()
                    # Found a valid one, try to go lower
                    break # Break inner loop, go to next r
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

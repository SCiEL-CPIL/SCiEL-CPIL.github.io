import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Generate the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j,k), gamma=(i,k)
    # Note: gamma index is k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def solve_for_W(U, V, T):
        # T[alpha, beta, gamma] = sum_c U[c, alpha] V[c, beta] W[c, gamma]
        # For each gamma, T[:, :, gamma] = sum_c (U[:, c] * V[:, c]) W[c, gamma]
        # Let M_gamma = T[:, :, gamma].reshape(-1, 1)
        # Let A = np.stack([U[:, c] * V[:, c] for c in range(r)], axis=1)
        # Then M = A @ W[:, gamma]
        r = U.shape[0]
        W = np.zeros((r, p*n))
        # A is (n*m*m*p, r)
        A = np.zeros((n*m, m*p, r))
        for c in range(r):
            A[:, :, c] = U[c, :].reshape(n, m)[:, :, np.newaxis] * V[c, :].reshape(m, p)[np.newaxis, :, :]
            # Wait, U[c] is (n*m), V[c] is (m*p)
            # U[c, alpha] where alpha = i*m+j
            # V[c, beta] where beta = j*p+k
            # Product is U[c, i*m+j] * V[c, j*p+k]
            # This is not a simple outer product of vectors of length n*m and m*p.
            # It's a specific structure.
            pass
        
        # Let's reshape T to (n*m*m*p, p*n)
        T_reshaped = T.reshape(n*m*m*p, p*n)
        
        # Construct A matrix of shape (n*m*m*p, r)
        A = np.zeros((n*m*m*p, r))
        for c in range(r):
            Uc = U[c, :].reshape(n, m)
            Vc = V[c, :].reshape(m, p)
            # We need to fill A[:, c] with Uc[i, j] * Vc[j, k] for all i, j, k
            # Index in A is (i*m+j) * (m*p) + (j*p+k) ? No.
            # T is indexed by alpha, beta.
            # alpha = i*m+j, beta = j*p+k.
            # So A[alpha * (m*p) + beta, c] = Uc[i, j] * Vc[j, k]
            for i in range(n):
                for j in range(m):
                    alpha = i * m + j
                    for k in range(p):
                        beta = j * p + k
                        idx = alpha * (m*p) + beta
                        A[idx, c] = Uc[i, j] * Vc[j, k]
        
        # Solve A @ W = T_reshaped
        # W is (r, p*n)
        # A is (N, r), T_reshaped is (N, p*n)
        # A @ W = T_reshaped
        # W = pinv(A) @ T_reshaped
        try:
            W = np.linalg.lstsq(A, T_reshaped, rcond=None)[0]
        except:
            return None
            
        return W

    def solve_for_U(V, W, T):
        r = V.shape[0]
        T_reshaped = T.reshape(n*m*m*p, p*n)
        # T[alpha, beta, gamma] = sum_c U[c, alpha] V[c, beta] W[c, gamma]
        # For each alpha, T[alpha, :, :] = sum_c U[c, alpha] (V[:, c] * W[:, c])
        # Let B = np.stack([V[:, c] * W[:, c] for c in range(r)], axis=1)
        # B is (m*p*p*n, r)
        # T_alpha = T[alpha, :, :].reshape(-1, 1)
        # T_alpha = B @ U[:, alpha]
        
        B = np.zeros((m*p*p*n, r))
        for c in range(r):
            Vc = V[c, :].reshape(m, p)
            Wc = W[c, :].reshape(p, n)
            for j in range(m):
                for k in range(p):
                    beta = j * p + k
                    for i in range(n):
                        gamma = k * n + i
                        idx = beta * (p*n) + gamma
                        B[idx, c] = Vc[j, k] * Wc[i, k] # Wait, Wc is (p, n) so Wc[k, i]
                        # W index is k*n+i. So Wc[k, i] is correct.
        
        U = np.zeros((r, n*m))
        for i in range(n):
            for j in range(m):
                alpha = i * m + j
                # Extract T[alpha, :, :]
                T_alpha = T[alpha, :, :].reshape(-1, 1)
                try:
                    U_col = np.linalg.lstsq(B, T_alpha, rcond=None)[0]
                    U[:, alpha] = U_col
                except:
                    return None
        return U

    def solve_for_V(U, W, T):
        r = U.shape[0]
        T_reshaped = T.reshape(n*m*m*p, p*n)
        # T[alpha, beta, gamma] = sum_c U[c, alpha] V[c, beta] W[c, gamma]
        # For each beta, T[:, beta, :] = sum_c V[c, beta] (U[:, c] * W[:, c])
        
        C = np.zeros((n*m*p*n, r))
        for c in range(r):
            Uc = U[c, :].reshape(n, m)
            Wc = W[c, :].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    alpha = i * m + j
                    for k in range(p):
                        gamma = k * n + i
                        idx = alpha * (p*n) + gamma # This indexing is tricky.
                        # T is (alpha, beta, gamma).
                        # We want to group by beta.
                        # T[:, beta, :] has shape (n*m, p*n).
                        # Flattened: (n*m*p*n,).
                        # Index in flattened T for (alpha, beta, gamma) is alpha * (m*p*p*n) + beta * (p*n) + gamma.
                        # So for fixed beta, we take T[:, beta, :].
                        pass
        
        # Let's just use the full tensor and solve for V by fixing U and W.
        # T = sum_c U[:, c, None, None] * V[None, c, :, None] * W[None, None, c, :]?
        # No, indices are mixed.
        
        # Let's stick to the loop for V.
        V = np.zeros((r, m*p))
        for j in range(m):
            for k in range(p):
                beta = j * p + k
                T_beta = T[:, beta, :].reshape(-1, 1) # Shape (n*m*p*n, 1)
                # T[alpha, beta, gamma] = sum_c U[c, alpha] V[c, beta] W[c, gamma]
                # Let D = U * W (element-wise product over c)
                # D[alpha, gamma, c] = U[c, alpha] * W[c, gamma]
                # T_beta[alpha, gamma] = sum_c D[alpha, gamma, c] V[c, beta]
                # So T_beta = D @ V[:, beta]
                
                D = np.zeros((n*m, p*n, r))
                for c in range(r):
                    Uc = U[c, :].reshape(n, m)
                    Wc = W[c, :].reshape(p, n)
                    for i in range(n):
                        for j_prime in range(m):
                            alpha = i * m + j_prime
                            for k_prime in range(p):
                                gamma = k_prime * n + i
                                D[alpha, gamma, c] = Uc[i, j_prime] * Wc[k_prime, i]
                
                D_reshaped = D.reshape(n*m*p*n, r)
                try:
                    V_col = np.linalg.lstsq(D_reshaped, T_beta, rcond=None)[0]
                    V[:, beta] = V_col
                except:
                    return None
        return V

    def check_solution(U, V, W):
        # Check if U, V, W are close to integer or half-integer
        def is_half_int(x):
            return np.allclose(x * 2, np.round(x * 2))
        
        if not (is_half_int(U) and is_half_int(V) and is_half_int(W)):
            return False
            
        # Check exact decomposition
        # T_calc[alpha, beta, gamma] = sum_c U[c, alpha] V[c, beta] W[c, gamma]
        # Compare with T
        # We only need to check the non-zero entries of T
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    val = 0.0
                    for c in range(U.shape[0]):
                        val += U[c, alpha] * V[c, beta] * W[c, gamma]
                    if abs(val - 1.0) > 1e-9:
                        error += abs(val - 1.0)
        return error < 1e-9

    # Try to find a solution with rank < 48
    # Start with rank 40 and decrease
    # Or start with 48 and try to reduce
    # Given time limit, try a few ranks
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    
    # Initialize with schoolbook
    U = np.zeros((rank_schoolbook, n*m))
    V = np.zeros((rank_schoolbook, m*p))
    W = np.zeros((rank_schoolbook, p*n))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[col, i*m+j] = 1
                V[col, j*p+k] = 1
                W[col, k*n+i] = 1
                col += 1
                
    # Try to improve
    # We can try random restarts with lower rank
    for r in range(40, 48):
        if time.time() > 110: break
        for _ in range(10):
            if time.time() > 115: break
            U_r = np.random.randn(r, n*m)
            V_r = np.random.randn(r, m*p)
            W_r = np.random.randn(r, p*n)
            
            for _ in range(20):
                W_r = solve_for_W(U_r, V_r, T)
                if W_r is None: continue
                U_r = solve_for_U(V_r, W_r, T)
                if U_r is None: continue
                V_r = solve_for_V(U_r, W_r, T)
                if V_r is None: continue
                
            if check_solution(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break
        if best_rank < 48:
            break
            
    if best_U is None:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


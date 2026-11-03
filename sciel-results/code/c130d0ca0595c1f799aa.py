import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Precompute indices for the sparse tensor
    # I = i*m + j, J = j*p + k, K = k*n + i
    # T[I, J, K] = 1 if i from I == i from K, j from I == j from J, k from J == k from K
    # Actually, for matmul, T[I, J, K] is 1 if the indices are consistent.
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    # Consistency:
    # i_I = I // m, j_I = I % m
    # j_J = J // p, k_J = J % p
    # k_K = K // n, i_K = K % n
    # Match if i_I == i_K and j_I == j_J and k_J == k_K
    
    # We don't need to build the full sparse matrix if we can check on the fly.
    
    min_rank = max(n * m, m * p, p * n)
    # Lower bound is actually max(n*m, m*p, p*n) is NOT correct for matmul.
    # Lower bound is roughly (n*m*p) / (max dimension)?
    # For 4x5x5, rank is likely around 40-60?
    
    # Try ranks from a reasonable lower bound
    # Let's try from 20 to 60
    
    for R in range(20, 61):
        np.random.seed(42 + R)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(100):
            # Solve for U
            # Target T[I, J, K] is 1 if consistent, else 0.
            # Equation: sum_r U[I, r] V[J, r] W[K, r] = T[I, J, K]
            # For fixed J, K, we solve for U[I, :]?
            # No, we solve for all U at once?
            # This is a linear system for each I?
            # sum_r U[I, r] (V[J, r] W[K, r]) = T[I, J, K]
            # This is not a simple linear system for U[I] because J and K vary.
            
            # Standard ALS for tensor:
            # Fix V, W.
            # For each I, we want U[I, :] such that:
            # sum_{J, K} (sum_r U[I, r] V[J, r] W[K, r] - T[I, J, K])^2 is minimized?
            # No, the error is sum_{I, J, K} ( ... )^2.
            # For fixed I, the term is sum_{J, K} (sum_r U[I, r] A_{I, J, K, r} - T[I, J, K])^2?
            # This is complex.
            
            # Simpler ALS:
            # For each I, we can treat the equation as:
            # sum_r U[I, r] B_{I, r} = C_I?
            # No.
            
            # Let's use the standard update:
            # U[I, :] = argmin_U sum_{J, K} (sum_r U[I, r] V[J, r] W[K, r] - T[I, J, K])^2
            # This is a least squares problem for U[I, :].
            # Let M_{J, K, r} = V[J, r] W[K, r].
            # We want sum_r U[I, r] M_{J, K, r} approx T[I, J, K].
            # This is equivalent to:
            # (V * W) @ U[I, :]^T approx T[I, :, :]
            # Where V*W is a tensor?
            
            # Vectorized:
            # Reshape V to (p, m, R)? No.
            # V is (m*p, R). J = j*p + k.
            # W is (p*n, R). K = k*n + i.
            
            # For a fixed I (i, j), we have T[I, J, K] non-zero only for specific J, K.
            # J must have j_J = j. So J = j*p + k for k in 0..p-1.
            # K must have i_K = i. So K = k*n + i for k in 0..p-1.
            # And k_J = k_K. So for each k, J = j*p + k, K = k*n + i.
            # So for fixed I, there are p pairs (J, K).
            
            # Equation for U[I, :]:
            # sum_{k=0}^{p-1} (sum_r U[I, r] V[j*p+k, r] W[k*n+i, r] - 1)^2 = 0
            # This is a system of p equations for R unknowns?
            # No, it's a single scalar equation per k?
            # We want sum_r U[I, r] (V[j*p+k, r] W[k*n+i, r]) = 1 for all k.
            
            # Let A_{I, k, r} = V[j*p+k, r] W[k*n+i, r].
            # We want sum_r U[I, r] A_{I, k, r} = 1 for all k.
            # This is a linear system: A_I U[I, :]^T = 1.
            # A_I is p x R.
            
            A_u = np.zeros((n * m, p, R))
            for i in range(n):
                for j in range(m):
                    I = i * m + j
                    for k in range(p):
                        J = j * p + k
                        K = k * n + i
                        A_u[I, k, :] = V[J, :] * W[K, :]
            
            # Solve A_u[I] @ U[I, :]^T = 1 (vector of ones of length p)
            # U[I, :] = (A_u[I] @ A_u[I]^T)^-1 A_u[I] @ 1
            # Or use lstsq.
            for I in range(n * m):
                # A_u[I] is p x R. We want x (R x 1) such that A_u[I] x = 1 (p x 1).
                # If p > R, overdetermined. If p < R, underdetermined.
                # Use lstsq.
                sol, residuals, rank, sv = np.linalg.lstsq(A_u[I], np.ones(p), rcond=None)
                U[I, :] = sol
            
            # Similarly for V and W.
            # For V[J, :] (j, k):
            # Sum over i.
            # J = j*p + k.
            # I must have j_I = j. I = i*m + j.
            # K must have k_K = k. K = k*n + i.
            # i_I = i_K = i.
            # So for each i, I = i*m + j, K = k*n + i.
            # Equation: sum_r V[J, r] U[I, r] W[K, r] = 1.
            
            A_v = np.zeros((m * p, n, R))
            for j in range(m):
                for k in range(p):
                    J = j * p + k
                    for i in range(n):
                        I = i * m + j
                        K = k * n + i
                        A_v[J, i, :] = U[I, :] * W[K, :]
                        
            for J in range(m * p):
                sol, residuals, rank, sv = np.linalg.lstsq(A_v[J], np.ones(n), rcond=None)
                V[J, :] = sol
            
            # For W[K, :] (k, i):
            # Sum over j.
            # K = k*n + i.
            # I must have i_K = i. I = i*m + j.
            # J must have k_J = k. J = j*p + k.
            # j_I = j_J = j.
            # So for each j, I = i*m + j, J = j*p + k.
            # Equation: sum_r W[K, r] U[I, r] V[J, r] = 1.
            
            A_w = np.zeros((p * n, m, R))
            for k in range(p):
                for i in range(n):
                    K = k * n + i
                    for j in range(m):
                        I = i * m + j
                        J = j * p + k
                        A_w[K, j, :] = U[I, :] * V[J, :]
                        
            for K in range(p * n):
                sol, residuals, rank, sv = np.linalg.lstsq(A_w[K], np.ones(m), rcond=None)
                W[K, :] = sol
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        val = np.dot(U[I, :], V[J, :] * W[K, :])
                        err += (val - 1.0) ** 2
                        
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_error < 1e-6:
                break
                
        if best_error < 1e-4:
            # Round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Verify
            is_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        val = np.dot(U_int[I, :], V_int[J, :] * W_int[K, :])
                        if abs(val - 1.0) > 1e-9:
                            is_exact = False
                            break
                    if not is_exact: break
                if not is_exact: break
                
            if is_exact:
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U[I][col] = 1
                V[J][col] = 1
                W[K][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

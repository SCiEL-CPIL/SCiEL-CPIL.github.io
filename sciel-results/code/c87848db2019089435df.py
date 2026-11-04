import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x7 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = get_schoolbook()
    best_rank = n * m * p

    # Tensor T construction
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def try_rank(R):
        if time.time() > deadline:
            return None
        
        for restart in range(5):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(100):
                if time.time() > deadline:
                    return None
                
                # Update U
                N = np.zeros((R, m * p * p * n))
                for b in range(m * p):
                    for c in range(p * n):
                        idx = b * (p * n) + c
                        N[:, idx] = V[:, 0] * W[:, 0] # Placeholder, need to fill correctly
                        # Actually, N[r, idx] = V[b, r] * W[c, r]
                        # This loop is inefficient. Let's use broadcasting.
                
                # Efficient N construction
                # V is (m*p, R), W is (p*n, R)
                # We want N[r, b*p*n + c] = V[b, r] * W[c, r]
                # Let's reshape V and W to (m*p, 1, R) and (1, p*n, R)
                V_reshaped = V.T.reshape(R, m * p, 1)
                W_reshaped = W.T.reshape(R, 1, p * n)
                N = (V_reshaped * W_reshaped).reshape(R, m * p * p * n)
                
                T_flat = T.reshape(n * m, m * p * p * n)
                
                try:
                    Nt = N.T
                    NtN = Nt @ N
                    NtN += 1e-8 * np.eye(R)
                    U = T_flat @ Nt @ np.linalg.inv(NtN)
                except np.linalg.LinAlgError:
                    pass
                
                # Update V
                U_reshaped = U.T.reshape(R, n * m, 1)
                W_reshaped = W.T.reshape(R, 1, p * n)
                N = (U_reshaped * W_reshaped).reshape(R, n * m * p * n)
                
                T_flat = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
                
                try:
                    Nt = N.T
                    NtN = Nt @ N
                    NtN += 1e-8 * np.eye(R)
                    V = T_flat @ Nt @ np.linalg.inv(NtN)
                except np.linalg.LinAlgError:
                    pass
                
                # Update W
                U_reshaped = U.T.reshape(R, n * m, 1)
                V_reshaped = V.T.reshape(R, 1, m * p)
                N = (U_reshaped * V_reshaped).reshape(R, n * m * m * p)
                
                T_flat = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
                
                try:
                    Nt = N.T
                    NtN = Nt @ N
                    NtN += 1e-8 * np.eye(R)
                    W = T_flat @ Nt @ np.linalg.inv(NtN)
                except np.linalg.LinAlgError:
                    pass
            
            T_approx = np.zeros_like(T)
            for r in range(R):
                T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            residual = np.linalg.norm(T - T_approx)
            
            if residual < 1e-6:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_approx_rounded = np.zeros_like(T)
                for r in range(R):
                    T_approx_rounded += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                
                if np.allclose(T, T_approx_rounded, atol=1e-9):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
        
        return None

    for R in range(55, 40, -1):
        result = try_rank(R)
        if result is not None:
            return result
    
    return best_result
# EVOLVE-BLOCK-END

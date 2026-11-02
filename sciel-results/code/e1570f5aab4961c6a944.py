# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Build the full tensor for verification
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        return T
    
    T = build_tensor(n, m, p)
    
    def verify_decomp(U, V, W, n, m, p, T, tol=1e-9):
        """Check if decomposition is exact."""
        nm, m_p, p_n = n*m, m*p, p*n
        r = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Compute predicted tensor
        pred = np.zeros((nm, m_p, p_n))
        for r_idx in range(r):
            pred += np.einsum('a,b,c->abc', U_mat[:, r_idx], V_mat[:, r_idx], W_mat[:, r_idx])
        
        return np.allclose(pred, T, atol=tol)

    # Use ALS to find low rank decomposition
    def als_decompose(T, rank, max_iters=100, tol=1e-10):
        """
        Alternating Least Squares for Tensor Decomposition.
        T is (I, J, K).
        Returns U (I, R), V (J, R), W (K, R).
        """
        I, J, K = T.shape
        R = rank
        
        # Initialize
        U = np.random.randn(I, R) * 0.1
        V = np.random.randn(J, R) * 0.1
        W = np.random.randn(K, R) * 0.1
        
        for _ in range(max_iters):
            # Update U
            # T_I = U Z^T
            Z = np.zeros((R, J * K))
            for r in range(R):
                Z[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            T_I = T.reshape(I, J * K)
            U_new, _, _, _ = np.linalg.lstsq(Z, T_I.T, rcond=None)
            U = U_new.T
            
            # Update V
            # T_J = V Q^T
            Q = np.zeros((R, I * K))
            for r in range(R):
                Q[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
            T_J = T.transpose(1, 0, 2).reshape(J, I * K)
            V_new, _, _, _ = np.linalg.lstsq(Q, T_J.T, rcond=None)
            V = V_new.T
            
            # Update W
            # T_K = W P^T
            P = np.zeros((R, I * J))
            for r in range(R):
                P[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            T_K = T.transpose(2, 0, 1).reshape(K, I * J)
            W_new, _, _, _ = np.linalg.lstsq(P, T_K.T, rcond=None)
            W = W_new.T
            
        return U, V, W

    # Try to find lower rank decomposition
    np.random.seed(42)
    
    # Try ranks 59 down to 45
    for target_rank in range(59, 44, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_restarts = 5
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            U_sol, V_sol, W_sol = als_decompose(T, target_rank, max_iters=50)
            
            # Check error
            pred = np.zeros((n*m, m*p, p*n))
            for r_idx in range(target_rank):
                pred += np.einsum('a,b,c->abc', U_sol[:, r_idx], V_sol[:, r_idx], W_sol[:, r_idx])
            
            error = np.linalg.norm(pred - T)
            
            if error < 1e-8:
                # Round to nearest 0.5
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                # Verify rounded
                if verify_decomp(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p, T):
                    best_rank = target_rank
                    best_U = U_round.tolist()
                    best_V = V_round.tolist()
                    best_W = W_round.tolist()
                    found = True
                    break
        if found:
            break

    # Final verification of best solution
    if best_rank < rank_schoolbook:
        if verify_decomp(best_U, best_V, best_W, n, m, p, T):
            return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return schoolbook as fallback
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from low to high, use ALS to find exact decomposition
    # Start from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(6,30,20) = 30
    # But actual rank might be higher
    
    best_result = None
    
    # Try various ranks
    for rank in [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 48, 50, 55, 60]:
        if time.time() > deadline - 30:
            break
        
        # Initialize with schoolbook-like structure
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize: use first min(rank, n*m*p) schoolbook terms
        nschool = min(rank, n*m*p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
            if col >= rank:
                break
            if col >= rank:
                break
        
        # Add some random initialization for remaining columns
        for c in range(col, rank):
            U[:, c] = np.random.randn(n*m) * 0.1
            V[:, c] = np.random.randn(m*p) * 0.1
            W[:, c] = np.random.randn(p*n) * 0.1
        
        # ALS iterations
        for it in range(2000):
            if time.time() > deadline - 30:
                break
            
            # Update U: for each row i of U, minimize ||T[i,:,:] - U[i,:] @ (V ⊙ W)||
            # T[i,:,:] is a (m*p) x (p*n) matrix
            # (V ⊙ W) is element-wise product, shape (m*p, p*n, rank)
            # We want: T[i] ≈ U[i,:] @ sum_c U[i,c] * (V[:,c] * W[:,c])
            # So: U[i,:] should be the solution to: T[i] = U[i,:] @ M where M[c, (a,b)] = V[a,c]*W[b,c]
            
            # Build M: shape (rank, m*p, p*n)
            M = np.einsum('ac,bc->abc', V, W)  # M[a,b,c] = V[a,c]*W[b,c]
            # M has shape (m*p, p*n, rank)
            # For each i: T[i, :, :] (shape m*p x p*n) ≈ U[i,:] @ M.T (shape rank x m*p x p*n flattened)
            # Actually: sum_c U[i,c] * M[:,:,c] = T[i,:,:]
            # Reshape M to (m*p*p*n, rank)
            M_flat = M.transpose(2, 0, 1).reshape(rank, m*p*p*n)
            T_flat = T.reshape(n*m, m*p*p*n)
            
            # Solve: T_flat[i] ≈ U[i,:] @ M_flat.T
            # U[i,:] = T_flat[i] @ (M_flat.T @ M_flat)^{-1}
            # Use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V: for each row a of V
            # T[:, a, :] ≈ (U ⊙ W) contracted... 
            # sum_c U[i,c]*V[a,c]*W[b,c] = T[i,a,b] where a=j*p+k, b=k'*n+i'
            # For fixed a: sum_c V[a,c] * (U[:,c] * W[:,c])[i] * ... 
            # Actually: for fixed a, T[:, a, :] (shape n*m x p*n) ≈ sum_c V[a,c] * outer(U[:,c], W[:,c])
            # Reshape: T[:, a, :] as vector of length n*m*p*n
            # M_V[c] = outer(U[:,c], W[:,c]) flattened, shape (n*m*p*n,)
            M_V = np.einsum('ic,kc->ick', U, W)  # shape (n*m, p*n, rank)
            M_V_flat = M_V.transpose(2, 0, 1).reshape(rank, n*m*p*n)
            T_V_flat = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)  # T[a, i, b]
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_V_flat.T, T_V_flat.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Update W: for each row b of W
            # T[:, :, b] ≈ sum_c W[b,c] * outer(U[:,c], V[:,c])
            M_W = np.einsum('ic,ac->iac', U, V)  # shape (n*m, m*p, rank)
            M_W_flat = M_W.transpose(2, 0, 1).reshape(rank, n*m*m*p)
            T_W_flat = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_W_flat.T, T_W_flat.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check convergence
            residual = np.linalg.norm(T - np.einsum('ic,ac,kc->iak', U, V, W))
            if residual < 1e-8:
                break
        
        # Check if we got an exact solution
        residual = np.linalg.norm(T - np.einsum('ic,ac,kc->iak', U, V, W))
        if residual < 1e-6:
            # Try to round to integers or half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            residual_round = np.linalg.norm(T - np.einsum('ic,ac,kc->iak', U_round, V_round, W_round))
            if residual_round < 1e-6:
                U, V, W = U_round, V_round, W_round
            
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
        elif residual < 1e-4:
            # Close but not exact, try more iterations
            for it2 in range(5000):
                if time.time() > deadline - 30:
                    break
                M = np.einsum('ac,bc->abc', V, W)
                M_flat = M.transpose(2, 0, 1).reshape(rank, m*p*p*n)
                T_flat = T.reshape(n*m, m*p*p*n)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                M_V = np.einsum('ic,kc->ick', U, W)
                M_V_flat = M_V.transpose(2, 0, 1).reshape(rank, n*m*p*n)
                T_V_flat = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V_flat.T, T_V_flat.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                M_W = np.einsum('ic,ac->iac', U, V)
                M_W_flat = M_W.transpose(2, 0, 1).reshape(rank, n*m*m*p)
                T_W_flat = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W_flat.T, T_W_flat.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                residual = np.linalg.norm(T - np.einsum('ic,ac,kc->iak', U, V, W))
                if residual < 1e-8:
                    break
            
            residual = np.linalg.norm(T - np.einsum('ic,ac,kc->iak', U, V, W))
            if residual < 1e-6:
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                residual_round = np.linalg.norm(T - np.einsum('ic,ac,kc->iak', U_round, V_round, W_round))
                if residual_round < 1e-6:
                    U, V, W = U_round, V_round, W_round
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END

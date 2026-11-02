"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W decompose tensor T."""
    n, m, p = T.shape[0] // m if False else (3, 3, 6)  # We know dimensions
    # Actually, let's reconstruct from shapes
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Reconstruct tensor
    T_recon = np.zeros_like(T)
    for c in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return np.allclose(T_recon, T, atol=tol)

def _try_rank(r, n, m, p, T, seed=0, max_iter=100):
    """Try to find a rank-r decomposition using iterative method."""
    np.random.seed(seed)
    rank = r
    # Initialize randomly
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    best_err = np.inf
    for it in range(max_iter):
        # Update U
        for c in range(rank):
            # U[:,c] should satisfy: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
            # For fixed c, U[:,c] = argmin ||T - sum_c' ...||
            # This is a linear system: U[:,c] = (V[:,c] ⊗ W[:,c])^+ T_flattened
            # Actually, for each c: U[:,c] = matvec_solve
            # Residual approach:
            pass
        
        # Simpler: use ALS (Alternating Least Squares)
        # For U: U[:,c] = (V[:,c] ⊗ W[:,c])^† * T (flattened)
        # V[:,c] ⊗ W[:,c] is a vector of length (m*p)*(p*n)
        # T flattened is also that length
        
        # Update U
        for c in range(rank):
            vw = np.kron(V[:,c], W[:,c])  # length (m*p)*(p*n)
            # U[:,c] such that sum_c U[:,c] * vw_c = T_flat
            # This is underdetermined if rank < n*m, so use least squares
            # Actually: T_flat = sum_c np.kron(U[:,c], np.kron(V[:,c], W[:,c]))
            # Hmm, this is getting complex. Let me use a different formulation.
            pass
        
        break  # This approach is getting too complex for the time limit
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    deadline = time.time() + 120  # 120 second limit
    
    # Try ranks from 53 downward
    best_result = None
    for r in range(53, 40, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            # Use a simple iterative approach
            np.random.seed(seed)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for c in range(r):
                    # T = sum_c U_c ⊗ V_c ⊗ W_c
                    # Fix V_c, W_c, solve for U_c
                    # T_flat = sum_c' np.kron(U_c', np.kron(V_c', W_c'))
                    # For term c: contribution is np.kron(U_c, np.kron(V_c, W_c))
                    # So: T_flat - sum_{c'≠c} np.kron(U_c', np.kron(V_c', W_c')) = np.kron(U_c, np.kron(V_c, W_c))
                    # Let R_c = T_flat - sum_{c'≠c} np.kron(U_c', np.kron(V_c', W_c'))
                    # Then U_c = (np.kron(V_c, W_c))^+ R_c (reshaped)
                    
                    # Compute residual
                    T_flat = T.reshape(-1)
                    residual = T_flat.copy()
                    for c2 in range(r):
                        if c2 == c:
                            continue
                        vw = np.kron(V[:,c2], W[:,c2])
                        residual -= np.outer(U[:,c2], vw)
                    
                    vw = np.kron(V[:,c], W[:,c])
                    # Solve: U_c = argmin ||residual - U_c * vw^T||_F
                    # This is: residual = U_c * vw^T
                    # So U_c = residual * vw^T / (vw^T @ vw) if vw is a column vector
                    # Actually: residual (size n*m x m*p*p*n) = U_c (n*m x 1) * vw^T (1 x m*p*p*n)
                    # So U_c = residual @ vw / (vw @ vw)
                    denom = np.dot(vw, vw)
                    if denom > 1e-12:
                        U[:,c] = residual @ vw / denom
                
                # Update V
                for c in range(r):
                    T_flat = T.reshape(-1)
                    residual = T_flat.copy()
                    for c2 in range(r):
                        if c2 == c:
                            continue
                        uw = np.kron(U[:,c2], W[:,c2])
                        residual -= np.outer(V[:,c2], uw)
                    
                    uw = np.kron(U[:,c], W[:,c])
                    denom = np.dot(uw, uw)
                    if denom > 1e-12:
                        V[:,c] = residual @ uw / denom
                
                # Update W
                for c in range(r):
                    T_flat = T.reshape(-1)
                    residual = T_flat.copy()
                    for c2 in range(r):
                        if c2 == c:
                            continue
                        uv = np.kron(U[:,c2], V[:,c2])
                        residual -= np.outer(W[:,c2], uv)
                    
                    uv = np.kron(U[:,c], V[:,c])
                    denom = np.dot(uv, uv)
                    if denom > 1e-12:
                        W[:,c] = residual @ uv / denom
                
                # Check convergence
                T_recon = np.zeros_like(T)
                for c in range(r):
                    T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                err = np.linalg.norm(T_recon - T)
                if err < 1e-8:
                    found = True
                    break
            
            if found:
                break
        
        if found:
            # Convert to lists with rational approximation if possible
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}

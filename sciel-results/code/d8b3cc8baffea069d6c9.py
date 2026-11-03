import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    target_dim = n * m * p
    
    # Target tensor flattened
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We map (i, j, k) to index i*m*p + j*p + k? 
    # The problem defines U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The equation is sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1.
    # Let's flatten the indices (i,j,k) into a single index idx = i*(m*p) + j*p + k.
    # Total size n*m*p.
    
    T = np.zeros(target_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * (m * p) + j * p + k
                T[idx] = 1.0

    def verify(U, V, W):
        # U, V, W can be numpy arrays or lists of lists
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        S = np.zeros(target_dim, dtype=float)
        for c in range(rank):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * (m * p) + j * p + k
                        S[idx] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        if not np.allclose(S, T, atol=1e-9):
            return False
        
        # Check if entries are half-integers
        for mat in [U, V, W]:
            for val in mat.flatten():
                val_float = float(val)
                if abs(val_float * 2 - round(val_float * 2)) > 1e-9:
                    return False
        return True

    def try_rank(R, rng, max_iters=50):
        # Initialize U, V, W with random small integers
        U = rng.integers(-2, 3, size=(N, R)).astype(float)
        V = rng.integers(-2, 3, size=(M, R)).astype(float)
        W = rng.integers(-2, 3, size=(P, R)).astype(float)
        
        # Alternating Least Squares
        # We want to minimize || T - sum_c u_c v_c w_c ||^2
        # This is non-convex.
        
        for _ in range(max_iters):
            # Update U
            # For each c, u_c should be such that sum_c u_c (v_c w_c)^T approx T
            # This is not simple linear least squares because of the product.
            # We can fix V and W, and treat the term (v_c w_c) as a vector?
            # No, u_c is a vector.
            # The term is u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i].
            # This is a 3-way product.
            
            # A simpler approach: Random restarts with local search?
            # Given the complexity, maybe just try to find a solution by solving linear systems if we fix 2 matrices.
            
            # Let's try a different strategy: 
            # Fix V and W. Then the equation is linear in U.
            # For each idx (i,j,k), T[idx] = sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
            # This is not linear in U because U is indexed by (i,j) and the sum is over c.
            # Wait, U[i*m+j, c] is the variable.
            # For a fixed (i,j), the equation is:
            # sum_k T[i,j,k] is not right.
            # The equation is for each (i,j,k):
            # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # If we fix V and W, we have a system of linear equations for U?
            # No, U[i*m+j, c] appears in multiple equations (for different k).
            # Specifically, for fixed i,j, we have p equations (one for each k).
            # And U[i*m+j, c] is a vector of length R.
            # Let U_ij be the vector U[i*m+j, :].
            # The equations are:
            # For k=0..p-1:
            # 1 = dot(U_ij, V_jk * W_ki)  (element-wise product of V and W vectors, then dot with U_ij)
            # This is a linear system for U_ij!
            # Matrix A_ij of size p x R.
            # A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
            # b_ij[k] = 1
            # Solve A_ij * U_ij = b_ij
            
            # So we can update U by solving p linear systems of size R x R (if p=R) or least squares.
            # Similarly for V and W.
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A_ij (p x R)
                    A = np.zeros((p, R))
                    for k in range(p):
                        for c in range(R):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b = np.ones(p)
                    # Solve A * U_ij = b
                    # Use least squares
                    U_ij, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = U_ij
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Build A_jk (n x R)
                    # Equations for i=0..n-1:
                    # 1 = dot(V_jk, U_ij * W_ki)
                    A = np.zeros((n, R))
                    for i in range(n):
                        for c in range(R):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b = np.ones(n)
                    V_jk, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = V_jk
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Build A_ki (m x R)
                    # Equations for j=0..m-1:
                    # 1 = dot(W_ki, U_ij * V_jk)
                    A = np.zeros((m, R))
                    for j in range(m):
                        for c in range(R):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b = np.ones(m)
                    W_ki, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = W_ki
            
            # Check convergence / exactness
            # Compute residual
            S = np.zeros(target_dim, dtype=float)
            for c in range(R):
                u = U[:, c]
                v = V[:, c]
                w = W[:, c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * (m * p) + j * p + k
                            S[idx] += u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            if np.allclose(S, T, atol=1e-5):
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                if verify(U_int, V_int, W_int):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None

    # Try ranks from 40 down to 30
    rng = np.random.default_rng(42)
    best_result = None
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    for R in range(40, 29, -1):
        for restart in range(5):
            if time.time() > deadline:
                break
            res = try_rank(R, rng, max_iters=20)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
        if time.time() > deadline:
            break
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((N, rank), dtype=int)
    V = np.zeros((M, rank), dtype=int)
    W = np.zeros((P, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

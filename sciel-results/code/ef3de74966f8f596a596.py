import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n
    total_constraints = n * m * p  # 100
    
    # Build index arrays for fast residual computation
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each (i,j,k), we need U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Precompute index triples
    ijks = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ijks.append((i*m+j, j*p+k, k*n+i))
    ijks = np.array(ijks)  # shape (100, 3)
    u_idx, v_idx, w_idx = ijks[:,0], ijks[:,1], ijks[:,2]
    
    # Schoolbook fallback
    def schoolbook():
        R = n*m*p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n*m*p
    
    def compute_residual(U, V, W):
        """Compute sum_r U[u_idx,r]*V[v_idx,r]*W[w_idx,r] for all 100 triples."""
        # For each r, compute contribution
        result = np.zeros(total_constraints)
        for r in range(U.shape[1]):
            contrib = U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            result += contrib
        return result
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        result = compute_residual(U, V, W)
        return np.allclose(result, 1.0, atol=tol)
    
    # Strategy: Use alternating least squares with structured initialization
    # Key insight: The tensor is T[i,j,k]=1. For fixed (j,k), we need sum_r U[i*m+j,r]*W[k*n+i,r] = 1/V[j*p+k,r] summed...
    # Better: Use the structure that for each j, the "slice" over i,k is all ones.
    
    # New structural approach: Parameterize using the fact that the multiplication
    # can be viewed as a sum of rank-1 updates. Use a "diagonal" W matrix structure.
    
    # Approach: Fix W to be block-diagonal with small blocks, solve for U,V.
    # Or: Use a randomized approach with good initialization.
    
    # Let's try: Initialize with a "partial schoolbook" that shares some terms.
    # For each j, the terms with same j can potentially be combined.
    
    # Actually, let's try a direct approach: use scipy's least squares on a 
    # reparameterized form.
    
    # Key idea: If we fix V and W, the problem becomes linear in U:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Group by (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for each k
    # This is a system: for fixed (i,j), we have p equations in R unknowns U[i*m+j, :]
    
    def try_als(R, n_restarts=5, time_per_restart=15.0):
        """Alternating least squares for fixed rank R."""
        t_start = time.time()
        
        for restart in range(n_restarts):
            if time.time() - t_start > time_per_restart * (restart + 1):
                break
            if time.time() > deadline - 10:
                break
            
            # Initialize
            rng = np.random.RandomState(restart * 12345 + 7)
            U = rng.randn(nm, R) * 0.5
            V = rng.randn(mp, R) * 0.5
            W = rng.randn(pn, R) * 0.5
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), solve linear system
                # sum_r U[i*m+j, r] * M_r = 1, where M_r = V[j*p+k, r]*W[k*n+i, r] for each k
                for i in range(n):
                    for j in range(m):
                        # Build the p x R matrix B where B[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        B = np.zeros((p, R))
                        for k in range(p):
                            B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve B @ U[i*m+j, :] = ones(p)
                        # Use least squares
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V: for each (j,k), solve
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, R))
                        for i in range(n):
                            B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W: for each (k,i), solve
                for k in range(p):
                    for i in range(n):
                        B = np.zeros((m, R))
                        for j in range(m):
                            B[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 0 or it == 199:
                    if verify(U, W, V if False else W, tol=1e-7):
                        # Double check
                        if verify(U, V, W, tol=1e-8):
                            return U, V, W, True
            
        return None, None, None, False
    
    # Search from low rank upward
    # Known: rank must be at least max(n*m, m*p, p*n) = max(20,25,20) = 25
    # But actual lower bound for matmul is higher. For 4x5x5, try from 40 to 90.
    
    for R in range(40, min(95, best_rank)):
        if time.time() > deadline - 15:
            break
        
        result_U, result_V, result_W, success = try_als(R, n_restarts=3, time_per_restart=20.0)
        
        if success:
            # Verify once more
            if verify(result_U, result_V, result_W, tol=1e-7):
                # Convert to half-integers if possible
                U_half = np.round(result_U * 2) / 2
                V_half = np.round(result_V * 2) / 2
                W_half = np.round(result_W * 2) / 2
                if verify(U_half, V_half, W_half, tol=1e-7):
                    best_U, best_V, best_W = U_half, V_half, W_half
                else:
                    best_U, best_V, best_W = result_U, result_V, result_W
                best_rank = R
                break  # Found minimal rank, stop
    
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

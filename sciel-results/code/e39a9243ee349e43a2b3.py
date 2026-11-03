import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20
    
    # The tensor T: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact (all products give 1)."""
        rank = U.shape[1]
        # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Also check half-integer constraint
        for mat in [U, V, W]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True
    
    def als_decompose(rank, n_iter=200, seed=42):
        """ALS optimization to find exact decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        # Precompute index mappings
        U_idx = [(i*m + j, i, j) for i in range(n) for j in range(m)]
        V_idx = [(j*p + k, j, k) for j in range(m) for k in range(p)]
        W_idx = [(k*n + i, k, i) for k in range(p) for i in range(n)]
        
        for it in range(n_iter):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), U[i*m+j,:] should satisfy
            # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # This is: U[i*m+j,:] @ diag(V[j*p+k,:]*W[k*n+i,:]) = 1 for all k
            # Stack: for fixed (i,j), we need U[i*m+j,:] such that
            # for each k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let M_k[r] = V[j*p+k,r] * W[k*n+i,r], then U[i*m+j,:] @ M_k = 1 for all k
            # So [M_0; M_1; ...; M_{p-1}] @ U[i*m+j,:]^T = [1,1,...,1]^T
            for (ui, i, j) in U_idx:
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                # Solve A_mat @ x = b in least squares
                x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                U[ui, :] = x
            
            # Update V: for each (j,k), V[j*p+k,:] such that
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            for (vi, j, k) in V_idx:
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                V[vi, :] = x
            
            # Update W: for each (k,i), W[k*n+i,:] such that
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            for (wi, k, i) in W_idx:
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                W[wi, :] = x
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-8:
            return U, V, W, max_err
        return None
    
    # Try ranks from low to high
    # Lower bound: for 2x3xp, rank >= 2*p (from 2x3x3 rank 7, so 2x3x10 >= 7*10/3 ~ 23.3, so >= 24)
    # Actually the exact lower bound is harder. Let's try from 20 upward.
    
    best_result = None
    
    for rank in range(20, 61):
        if time.time() > deadline:
            break
        
        result = als_decompose(rank, n_iter=100, seed=rank)
        if result is not None:
            U, V, W, err = result
            if verify(U, V, W):
                best_result = (rank, U, V, W)
                break
        else:
            # Try a few more seeds
            for seed in range(rank, rank+3):
                if time.time() > deadline:
                    break
                result = als_decompose(rank, n_iter=50, seed=seed)
                if result is not None:
                    U, V, W, err = result
                    if verify(U, V, W):
                        best_result = (rank, U, V, W)
                        break
            if best_result is not None:
                break
    
    if best_result is not None:
        rank, U, V, W = best_result
        # Convert to lists with half-integer rounding
        U_list = [[round(val * 2) / 2 for val in row] for row in U]
        V_list = [[round(val * 2) / 2 for val in row] for row in V]
        W_list = [[round(val * 2) / 2 for val in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

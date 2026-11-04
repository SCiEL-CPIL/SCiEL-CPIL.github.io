"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # The matrix multiplication tensor has 1s where indices match
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def als_decompose(target_rank, max_iters=200, tol=1e-10):
        """
        Alternating Least Squares for tensor decomposition.
        We want T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for all i,j,k.
        """
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Precompute index mappings
        U_idx = {}  # (i,j) -> row index in U
        V_idx = {}  # (j,k) -> row index in V
        W_idx = {}  # (k,i) -> row index in W
        for i in range(n):
            for j in range(m):
                U_idx[(i,j)] = i*m + j
        for j in range(m):
            for k in range(p):
                V_idx[(j,k)] = j*p + k
        for k in range(p):
            for i in range(n):
                W_idx[(k,i)] = k*n + i
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    row_u = U_idx[(i,j)]
                    # Build system: for each k, sum_c U[row_u,c] * V[V_idx[(j,k)],c] * W[W_idx[(k,i)],c] = 1
                    # Let M_k = V[V_idx[(j,k)]] * W[W_idx[(k,i)]] (element-wise product)
                    # Then sum_c U[row_u,c] * M_k[c] = 1 for each k
                    # This is: U[row_u] @ M^T = [1, 1, ..., 1]^T
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k] = V[V_idx[(j,k)]] * W[W_idx[(k,i)]]
                    # Solve U[row_u] @ M^T = ones(p)
                    try:
                        U[row_u] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    row_v = V_idx[(j,k)]
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i] = U[U_idx[(i,j)]] * W[W_idx[(k,i)]]
                    try:
                        V[row_v] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    row_w = W_idx[(k,i)]
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j] = U[U_idx[(i,j)]] * V[V_idx[(j,k)]]
                    try:
                        W[row_w] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[U_idx[(i,j)]], V[V_idx[(j,k)]] * W[W_idx[(k,i)]])
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < tol:
                break
        
        # Convert to list of lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if verify(U_list, V_list, W_list, target_rank):
            return U_list, V_list, W_list, target_rank
        return None
    
    def structured_search(target_rank, max_restarts=5):
        """Try ALS with multiple random restarts."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            result = als_decompose(target_rank)
            if result is not None:
                return result
        return None
    
    # Start with schoolbook as fallback
    U, V, W, rank = schoolbook()
    
    # Try to find lower rank decompositions using ALS
    # Lower bound: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    # Upper bound: n*m*p = 90
    for target_rank in range(30, 90):
        if time.time() > deadline:
            break
        
        result = structured_search(target_rank, max_restarts=3)
        if result is not None:
            U, V, W, rank = result
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

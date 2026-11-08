"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    
    T_target = _build_tensor(n, m, p)
    
    if not np.allclose(T_decomp, T_target, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _als_decompose(T, n, m, p, rank, max_iter=100, seed=0):
    """
    Alternating Least Squares for tensor decomposition.
    T is the target tensor (n*m, m*p, p*n).
    """
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for _ in range(max_iter):
        # Solve for U
        A = np.zeros((m*p * p*n, rank))
        B = np.zeros((m*p * p*n, n*m))
        
        idx = 0
        for j in range(m*p):
            for k in range(p*n):
                A[idx, :] = V[:, j] * W[:, k]
                B[idx, :] = T[:, j, k]
                idx += 1
                
        U_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        U = U_sol.T
        
        # Solve for V
        A = np.zeros((n*m * p*n, rank))
        B = np.zeros((n*m * p*n, m*p))
        idx = 0
        for i in range(n*m):
            for k in range(p*n):
                A[idx, :] = U[i, :] * W[k, :]
                B[idx, :] = T[i, :, k]
                idx += 1
        V_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        V = V_sol.T
        
        # Solve for W
        A = np.zeros((n*m * m*p, rank))
        B = np.zeros((n*m * m*p, p*n))
        idx = 0
        for i in range(n*m):
            for j in range(m*p):
                A[idx, :] = U[i, :] * V[j, :]
                B[idx, :] = T[i, j, :]
                idx += 1
        W_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        W = W_sol.T
        
    return U, V, W

def _try_rounding(U, V, W, n, m, p, rank, deadline):
    """Try to round the ALS solution to integer or half-integer values."""
    # Try different rounding strategies
    for scale in [1, 2]:
        U_rounded = np.round(U * scale) / scale
        V_rounded = np.round(V * scale) / scale
        W_rounded = np.round(W * scale) / scale
        
        def to_list(mat):
            result = []
            for row in mat:
                new_row = []
                for v in row:
                    # Check if close to integer or half-integer
                    if abs(v - round(v)) < 1e-6:
                        new_row.append(int(round(v)))
                    elif abs(2*v - round(2*v)) < 1e-6:
                        new_row.append(v)  # keep as float (half-integer)
                    else:
                        return None
                result.append(new_row)
            return result
        
        U_list = to_list(U_rounded)
        V_list = to_list(V_rounded)
        W_list = to_list(W_rounded)
        
        if U_list is not None and V_list is not None and W_list is not None:
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
    return None

def _structured_search(n, m, p, rank, deadline):
    """
    Structured search that exploits the sparsity of the matrix multiplication tensor.
    We look for decompositions where each term has a simple structure.
    """
    # For each rank, try to find a decomposition by solving a system of equations
    # The tensor has n*m*p = 45 non-zero entries
    # Each rank-r term contributes to at most r*n*m*p entries in the decomposition
    
    # We'll use a greedy approach: for each of the 45 non-zero entries,
    # try to assign it to one of the rank terms
    T = _build_tensor(n, m, p)
    
    # Get the list of non-zero entries
    non_zero_entries = []
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                if T[i,j,k] != 0:
                    non_zero_entries.append((i,j,k))
    
    # For a structured decomposition, each term (a,b,c) should satisfy
    # a[i,j] * b[j,k] * c[k,i] = 1 for the entries it covers
    
    # Try a simple structured approach: each term covers a subset of entries
    # where the indices have a specific relationship
    
    # This is a simplified structured search - we'll try to find terms
    # where the support is a "line" in the tensor
    
    best_result = None
    
    # Try different random structured initializations
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(seed)
        
        # Initialize with structured patterns
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Each column gets a random pattern
        for c in range(rank):
            # Random subset of non-zero positions
            u_support = set(np.random.choice(n*m, size=max(1, n*m//3), replace=False))
            v_support = set(np.random.choice(m*p, size=max(1, m*p//3), replace=False))
            w_support = set(np.random.choice(p*n, size=max(1, p*n//3), replace=False))
            
            for idx in u_support:
                U[idx, c] = 1.0
            for idx in v_support:
                V[idx, c] = 1.0
            for idx in w_support:
                W[idx, c] = 1.0
        
        # Run a few ALS iterations
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Solve for U
            A = np.zeros((m*p * p*n, rank))
            B = np.zeros((m*p * p*n, n*m))
            idx = 0
            for j in range(m*p):
                for k in range(p*n):
                    A[idx, :] = V[:, j] * W[:, k]
                    B[idx, :] = T[:, j, k]
                    idx += 1
            U_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
            U = U_sol.T
            
            # Solve for V
            A = np.zeros((n*m * p*n, rank))
            B = np.zeros((n*m * p*n, m*p))
            idx = 0
            for i in range(n*m):
                for k in range(p*n):
                    A[idx, :] = U[i, :] * W[k, :]
                    B[idx, :] = T[i, :, k]
                    idx += 1
            V_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
            V = V_sol.T
            
            # Solve for W
            A = np.zeros((n*m * m*p, rank))
            B = np.zeros((n*m * m*p, p*n))
            idx = 0
            for i in range(n*m):
                for j in range(m*p):
                    A[idx, :] = U[i, :] * V[j, :]
                    B[idx, :] = T[i, j, :]
                    idx += 1
            W_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
            W = W_sol.T
        
        # Try rounding
        result = _try_rounding(U, V, W, n, m, p, rank, deadline)
        if result:
            return result
    
    return None

def _find_decomposition(n, m, p, deadline):
    T = _build_tensor(n, m, p)
    
    # Try ranks from 30 up to 44
    # Start with lower ranks first since we want rank-minimal
    for r in range(30, 45):
        if time.time() > deadline - 10:
            break
        
        # First try structured search
        result = _structured_search(n, m, p, r, deadline)
        if result:
            return result
        
        # Then try random ALS
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            try:
                U, V, W = _als_decompose(T, n, m, p, r, max_iter=30, seed=seed)
                result = _try_rounding(U, V, W, n, m, p, r, deadline)
                if result:
                    return result
            except Exception:
                continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    # Try to find a better decomposition
    result = _find_decomposition(n, m, p, deadline)
    if result:
        return result
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END


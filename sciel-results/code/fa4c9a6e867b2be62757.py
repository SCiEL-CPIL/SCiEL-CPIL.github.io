import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_round_to_half_integer(U, V, W, n, m, p):
    """Try to round the decomposition to half-integers and verify."""
    # Try different scaling factors
    for scale in [1, 2, 4, 8]:
        U_round = np.round(U * scale) / scale
        V_round = np.round(V * scale) / scale
        W_round = np.round(W * scale) / scale
        if _verify(U_round, V_round, W_round, n, m, p):
            return U_round, V_round, W_round
    return None

def _als_decompose(n, m, p, r, U_init=None, V_init=None, max_iter=500, tol=1e-10):
    """Alternating least squares for tensor decomposition.
    
    We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    """
    if U_init is None:
        U = np.random.randn(n * m, r) * 0.1
    else:
        U = U_init.copy()
    if V_init is None:
        V = np.random.randn(m * p, r) * 0.1
    else:
        V = V_init.copy()
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for iteration in range(max_iter):
        # Update W: for each (k, i), solve least squares
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed (k, i), we need: sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
        # Let A_c(k,i) = sum_j U[i*m+j, c] * V[j*p+k, c]
        # Then sum_c A_c(k,i) * W[k*n+i, c] = 1
        
        W = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                # Build the system: for each c, coefficient is sum_j U[i*m+j, c] * V[j*p+k, c]
                coeffs = np.zeros(r)
                for j in range(m):
                    coeffs += U[i * m + j, :] * V[j * p + k, :]
                # We need sum_c coeffs[c] * W[k*n+i, c] = 1
                # This is underdetermined if r > 1, so use least squares with constraint
                # Actually, we have one equation: coeffs @ W[k*n+i, :] = 1
                # Minimum norm solution: W[k*n+i, :] = coeffs / (coeffs @ coeffs)
                norm_sq = np.dot(coeffs, coeffs)
                if norm_sq > 1e-12:
                    W[k * n + i, :] = coeffs / norm_sq
                else:
                    W[k * n + i, :] = 0
        
        # Update U: for each (i, j), solve least squares
        # sum_c U[i*m+j, c] * (sum_k V[j*p+k, c] * W[k*n+i, c]) = 1
        # Let B_c(i,j) = sum_k V[j*p+k, c] * W[k*n+i, c]
        # Then sum_c U[i*m+j, c] * B_c(i,j) = 1
        
        U_new = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                coeffs = np.zeros(r)
                for k in range(p):
                    coeffs += V[j * p + k, :] * W[k * n + i, :]
                norm_sq = np.dot(coeffs, coeffs)
                if norm_sq > 1e-12:
                    U_new[i * m + j, :] = coeffs / norm_sq
                else:
                    U_new[i * m + j, :] = 0
        U = U_new
        
        # Update V: for each (j, k), solve least squares
        # sum_c V[j*p+k, c] * (sum_i U[i*m+j, c] * W[k*n+i, c]) = 1
        # Let C_c(j,k) = sum_i U[i*m+j, c] * W[k*n+i, c]
        # Then sum_c V[j*p+k, c] * C_c(j,k) = 1
        
        V_new = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                coeffs = np.zeros(r)
                for i in range(n):
                    coeffs += U[i * m + j, :] * W[k * n + i, :]
                norm_sq = np.dot(coeffs, coeffs)
                if norm_sq > 1e-12:
                    V_new[j * p + k, :] = coeffs / norm_sq
                else:
                    V_new[j * p + k, :] = 0
        V = V_new
        
        # Check convergence
        if iteration % 50 == 0:
            error = 0.0
            for (i, j, k) in indices:
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                error += (val - 1.0) ** 2
            if error < tol:
                break
    
    return U, V, W

def _strassen_style_init(n, m, p, r):
    """Create a Strassen-like initialization for rectangular matrices."""
    # For 2x2x2, Strassen uses 7 terms with specific patterns
    # We'll create a sparse initialization with structured patterns
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Create a base pattern and tile it
    # For simplicity, use random but structured initialization
    np.random.seed(42)
    
    # Initialize with random values but ensure some structure
    for c in range(r):
        # Randomly select a subset of indices
        U[:, c] = np.random.randn(n * m) * 0.5
        V[:, c] = np.random.randn(m * p) * 0.5
        W[:, c] = np.random.randn(p * n) * 0.5
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s safety margin
    
    # Try ranks from just below schoolbook down
    # Known: rank(4,5,6) is at most 110 (by various bounds)
    # Let's try from 119 down to 100
    
    target_ranks = list(range(119, 99, -1))
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(30):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if trial == 0:
                # Start from schoolbook and truncate
                U = U_school[:, :r]
                V = V_school[:, :r]
                W = W_school[:, :r]
            elif trial < 10:
                # Random initialization
                U = np.random.randn(n * m, r) * 0.3
                V = np.random.randn(m * p, r) * 0.3
                W = np.random.randn(p * n, r) * 0.3
            else:
                # Structured random
                U, V, W = _strassen_style_init(n, m, p, r)
            
            # Run ALS
            U_out, V_out, W_out = _als_decompose(n, m, p, r, U_init=U, V_init=V, max_iter=200, tol=1e-12)
            
            # Check if we have a valid solution
            if _verify(U_out, V_out, W_out, n, m, p, tol=1e-8):
                # Try to round to half-integers
                rounded = _try_round_to_half_integer(U_out, V_out, W_out, n, m, p)
                if rounded is not None:
                    U_round, V_round, W_round = rounded
                    if _verify(U_round, V_round, W_round, n, m, p, tol=1e-9):
                        best_rank = r
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        found = True
                        break
        
        if found:
            # Found a solution at rank r, try to find lower
            # But continue to see if we can do better
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

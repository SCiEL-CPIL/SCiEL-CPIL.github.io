import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify if U, V, W form a valid decomposition."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for r in range(R):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - expected) > tol:
                    return False
    return True

def _alternating_minimization(n, m, p, R, max_iters=500, deadline=None):
    """
    Use alternating minimization to find a rank-R decomposition.
    For fixed V,W: solve for U via linear least squares
    For fixed U,W: solve for V via linear least squares  
    For fixed U,V: solve for W via linear least squares
    """
    np.random.seed(12345)
    
    U = np.random.randn(n*m, R) * 0.1
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Fix V and W, solve for U
        # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]).T = 1 for all k
        # So: A[i,j] @ U[i*m+j, :].T = ones(p)
        # where A[i,j,k,r] = V[j*p+k, r] * W[k*n+i, r]
        
        U_new = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                # Build matrix: p x R
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ U[i*m+j, :].T = ones(p)
                # Use least squares
                try:
                    sol = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U_new[i*m+j, :] = sol
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        U = U_new
        
        # Fix U and W, solve for V
        V_new = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                # Build matrix: n x R
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    V_new[j*p+k, :] = sol
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        V = V_new
        
        # Fix U and V, solve for W
        W_new = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                # Build matrix: m x R
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    sol = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    W_new[k*n+i, :] = sol
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        W = W_new
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-10:
            break
    
    # Final verification
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-8:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 105  # Safety margin
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # Try alternating minimization with various ranks
    for R in range(40, best_rank + 1):
        if time.time() > deadline:
            break
        
        result = _alternating_minimization(n, m, p, R, max_iters=200, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

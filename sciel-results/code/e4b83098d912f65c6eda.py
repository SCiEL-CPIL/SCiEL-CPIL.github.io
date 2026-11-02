"""Matrix multiplication 2x3x4 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    rank = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_polish(U, V, W, n, m, p, max_iter=1000, tol=1e-10):
    """
    Alternating Least Squares to minimize the error of the tensor decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Target: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    r = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    prev_err = float('inf')
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                # Solve for U[i*m+j, :]
                # Minimize sum_k (1 - sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c])^2
                # Let x = U[i*m+j, :]
                # y = 1
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A x = 1
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # Solve for V[j*p+k, :]
                # Minimize sum_i (1 - sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c])^2
                # Let x = V[j*p+k, :]
                # y = 1
                # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # Solve for W[k*n+i, :]
                # Minimize sum_j (1 - sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c])^2
                # Let x = W[k*n+i, :]
                # y = 1
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
            
    return U, V, W, prev_err

def _round_and_verify(U, V, W, n, m, p):
    """Try to round to nearest 0.5 and verify."""
    # Try rounding to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, n, m, p, tol=1e-9):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Search for lower ranks
    # Known lower bound is max(n*m, m*p, p*n) = 12
    # Try ranks from 12 upwards
    
    for target_rank in range(12, rank_sb):
        if time.time() > deadline - 10:
            break
        
        # Number of attempts depends on time remaining
        max_attempts = 100 if target_rank < 18 else 50
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            if found:
                break
            
            # Random initialization
            # Use small integers to help rounding
            U = np.random.randint(-2, 3, size=(n * m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, target_rank)) * 0.5
            
            # Polish with ALS
            U_p, V_p, W_p, err = _als_polish(U, V, W, n, m, p, max_iter=500)
            
            # Check if error is small enough to attempt rounding
            if err < 1e-4:
                result = _round_and_verify(U_p, V_p, W_p, n, m, p)
                if result:
                    U_r, V_r, W_r = result
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    found = True
                    break
        
        if found:
            break
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")

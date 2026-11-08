"""Matrix multiplication 3x5 by 5x5 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 140.0  # Safety margin before 150s limit

def _verify(U, V, W, n=3, m=5, p=5):
    """Verify that U, V, W form a valid decomposition of matrix multiplication.
    
    U: (n*m, rank)
    V: (m*p, rank)  
    W: (p*n, rank)
    
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    # Verify: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook(n=3, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _refine(U, V, W, n, m, p, max_iters=50, tol=1e-10):
    """Iteratively refine the decomposition using correct least-squares updates."""
    rank = U.shape[1]
    
    for _ in range(max_iters):
        # Update W
        W_new = np.zeros_like(W)
        for i in range(n):
            for k in range(p):
                # System: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
                # Matrix A is m x rank, where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve A w = 1
                try:
                    # Use least squares if rank > m, or direct solve if square/invertible
                    w, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=1e-15)
                    W_new[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    # Fallback to pinv
                    W_new[k * n + i, :] = np.linalg.pinv(A) @ np.ones(m)
        
        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # System: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # Matrix A is p x rank, where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                try:
                    u, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(p), rcond=1e-15)
                    U_new[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    U_new[i * m + j, :] = np.linalg.pinv(A) @ np.ones(p)
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # System: sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1 for all i
                # Matrix A is n x rank, where A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                
                try:
                    v, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(n), rcond=1e-15)
                    V_new[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    V_new[j * p + k, :] = np.linalg.pinv(A) @ np.ones(n)
        
        # Check convergence
        if (np.max(np.abs(W_new - W)) < tol and 
            np.max(np.abs(U_new - U)) < tol and 
            np.max(np.abs(V_new - V)) < tol):
            U, V, W = U_new, V_new, W_new
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Early exit if verification passes
        if _verify(U, V, W, n, m, p):
            break
            
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    np.random.seed(42)
    
    # Lower bound is max(n*m, m*p, p*n) = 25
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to best_rank - 1
    for rank in range(lower_bound, best_rank):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try multiple random initializations
        max_tries = 20
        found = False
        for attempt in range(max_tries):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Refine
            U, V, W = _refine(U, V, W, n, m, p, max_iters=100)
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                found = True
                break
        
        if found:
            # If we found a lower rank, we can stop or continue to see if we can go lower
            # Since we are iterating from low to high, the first valid rank found is the minimum
            # for this specific search path, but not necessarily global.
            # However, if we found rank R, we know R is valid.
            # We can break if we want the first found, but maybe a lower one exists?
            # The loop goes from lower_bound upwards. So the first one found is the lowest 
            # rank discovered so far.
            # We can break out of the rank loop if we found one, as we are going up.
            break

    # Convert to lists of lists with proper types
    result = {}
    for key in ['U', 'V', 'W']:
        arr = np.array(best[key])
        # Check if entries are close to integers or half-integers
        rounded = np.round(arr * 2) / 2
        if np.max(np.abs(arr - rounded)) < 1e-9:
            result[key] = rounded.tolist()
        else:
            result[key] = arr.tolist()
    
    return result

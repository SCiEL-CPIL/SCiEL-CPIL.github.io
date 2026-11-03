"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_exact(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is exact: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_for_rank(R, n, m, p, seed, max_iters=200, deadline=None):
    """
    Alternating Least Squares to find rank-R decomposition.
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    """
    np.random.seed(seed)
    
    # Initialize with sparse structured matrices
    U = np.random.randint(-1, 2, size=(n*m, R)) * 0.5  # values in {-0.5, 0, 0.5}
    V = np.random.randint(-1, 2, size=(m*p, R)) * 0.5
    W = np.random.randint(-1, 2, size=(p*n, R)) * 0.5
    
    # Ensure no row is all zeros
    for idx in range(n*m):
        if np.all(U[idx] == 0):
            U[idx, np.random.randint(R)] = 1.0
    for idx in range(m*p):
        if np.all(V[idx] == 0):
            V[idx, np.random.randint(R)] = 1.0
    for idx in range(p*n):
        if np.all(W[idx] == 0):
            W[idx, np.random.randint(R)] = 1.0
    
    best_residual = np.inf
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve for U given V, W
        # For each (i,j): sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1 for all k
        # This is a linear system in U[i,j,:]
        A_U = np.zeros((p, R))  # For each i,j, we have p equations
        b_U = np.ones(p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A_U[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A_U @ U[i*m+j, :] = b_U
                try:
                    U[i*m+j, :], residuals, rank_u, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
                except:
                    break
            
            if deadline is not None and time.time() > deadline:
                break
        
        # Solve for V given U, W
        # For each (j,k): sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r]) = 1 for all i
        A_V = np.zeros((n, R))
        b_V = np.ones(n)
        
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    A_V[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :], residuals, rank_v, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
                except:
                    break
            
            if deadline is not None and time.time() > deadline:
                break
        
        # Solve for W given U, V
        # For each (k,i): sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1 for all j
        A_W = np.zeros((m, R))
        b_W = np.ones(m)
        
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    A_W[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :], residuals, rank_w, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                except:
                    break
            
            if deadline is not None and time.time() > deadline:
                break
        
        # Check convergence
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    residual += (val - 1.0) ** 2
        
        if residual < best_residual:
            best_residual = residual
        
        if residual < 1e-12:
            # Found exact solution!
            return U, V, W, True
        
        # Periodic check
        if iteration % 10 == 0 and _check_exact(U, V, W, n, m, p, tol=1e-10):
            return U, V, W, True
    
    # Final check
    if _check_exact(U, V, W, n, m, p, tol=1e-10):
        return U, V, W, True
    
    return None, None, None, False

def _try_structured_rank(R, n, m, p, deadline):
    """Try multiple structured initializations for rank R."""
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1000, 1111, 2222]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        U, V, W, success = _als_for_rank(R, n, m, p, seed, max_iters=150, deadline=deadline)
        if success:
            return U, V, W
    return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # First, build schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook()
    
    # Try to find lower rank decompositions
    # Start with rank 25 (half of schoolbook)
    ranks_to_try = [25, 30, 35, 40, 45, 49]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        U, V, W = _try_structured_rank(R, n, m, p, deadline)
        if U is not None:
            # Verify one more time
            if _check_exact(U, V, W, n, m, p, tol=1e-10):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END

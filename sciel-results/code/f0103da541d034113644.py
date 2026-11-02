"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook-1 down to 20
    # If rank 24 is minimal, this will fail and we return 24.
    for rank in range(schoolbook_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                # If we found a lower rank, we might want to check even lower,
                # but given time constraints, we stop or continue quickly.
                continue
        except Exception:
            continue
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank using structured search."""
    np.random.seed(42)
    
    # Strategy: Try multiple random initializations with alternating least squares
    # Optimized for speed: fewer iterations, tighter checks
    num_attempts = 10
    max_iterations = 30
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Alternating minimization
        converged = False
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        # Solve A.T @ U_row = ones(p)
                        U_new[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V_new[j * p + k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W_new[k * n + i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                converged = True
                break
        
        if converged:
            # Try to snap to half-integers
            for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                U_snap = np.round(U * denom) / denom
                V_snap = np.round(V * denom) / denom
                W_snap = np.round(W * denom) / denom
                
                # Verify exactness
                max_err_snap = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_snap[i * m + j, :], V_snap[j * p + k, :] * W_snap[k * n + i, :])
                            max_err_snap = max(max_err_snap, abs(val - 1.0))
                
                if max_err_snap < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
    
    return None

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 75
    schoolbook_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank

    deadline = time.time() + 140.0  # safety margin

    # Try to find a lower-rank decomposition using structured search
    # For 3x5*5x5, known bounds: rank is between 3*5*5=75 (schoolbook) and 
    # Strassen-like improvements. Let's try random rank-64, rank-60, etc.
    
    # Actually, let's think about this more carefully.
    # The tensor T[i,j,k] = delta_{i,k} for multiplying A(3x5) * B(5x5) = C(3x5)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if we're looking at the (i,k) output and j is the inner index
    # In the factorization: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: W uses trace convention W[k*n+i, r] which means W is indexed by (k, i) -> row k*n+i
    
    # Let's try a systematic approach: use random matrices and try to solve for rank r
    # We'll use a structured approach with random initialization and alternating least squares
    
    def _try_rank(r, n, m, p, max_iters=200, seed=42):
        """Try to find rank-r decomposition using alternating least squares."""
        rng = np.random.RandomState(seed)
        
        # Initialize with random entries
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve least squares
            # U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:] for all k summed) should give T[i,j,k]
            # This is tricky because it's a 3-way product. Let's use a different approach.
            
            # Alternative: fix two, update one
            # For U[i*m+j, c]: we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # Let M_u = V[j*p+k,:] * W[k*n+i,:] element-wise, then U[i*m+j,:] @ M_u = ones(p)
            # Actually, let's define for fixed i,j:
            # For each k: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # Let B[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
            # Then U[i*m+j,:] @ B[i,j,:,:].T = ones(p)
            
            for i in range(n):
                for j in range(m):
                    # Build B matrix of shape (p, r)
                    B = np.zeros((p, r))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve U[i*m+j, :] @ B.T = ones(p)
                    # i.e., B @ U[i*m+j, :].T = ones(p)
                    try:
                        U[i * m + j, :], residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V: for each (j,k), sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :], residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W: for each (k,i), sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, r))
                    for j in range(m):
                        B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :], residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    except:
                        pass
            
            # Check convergence
            if it % 20 == 0:
                err = _compute_error(U, V, W, n, m, p)
                if err < 1e-6:
                    return U, V, W
        
        # Final check
        err = _compute_error(U, V, W, n, m, p)
        if err < 1e-4:
            return U, V, W
        return None
    
    def _compute_error(U, V, W, n, m, p):
        """Compute the error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def _schoolbook(n, m, p):
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
    
    # Try various ranks below schoolbook
    for r in [70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50]:
        if time.time() > deadline:
            break
        for seed in [42, 123, 456, 789, 111]:
            if time.time() > deadline:
                break
            result = _try_rank(r, n, m, p, max_iters=150, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists with integer/half-integer entries if possible
                U_list, V_list, W_list = _to_list_format(U, V, W)
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = r
                break
        if best_rank < schoolbook_rank:
            break
    
    # Convert to required format
    if best_rank < schoolbook_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": _to_list_format(best_U, best_V, best_W)[0], 
                "V": _to_list_format(best_U, best_V, best_W)[1], 
                "W": _to_list_format(best_U, best_V, best_W)[2]}

def _to_list_format(U, V, W):
    """Convert numpy arrays to list format."""
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return U_list, V_list, W_list

# EVOLVE-BLOCK-END

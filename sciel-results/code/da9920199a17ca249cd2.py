"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

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

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    # T[i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 if i,j,k match the tensor entry
    # The multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k (since (AB)[i,k] = sum_j A[i,j]B[j,k])
    # Actually the tensor for matrix mult is: T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the bilinear form)
    # Wait, let me reconsider. The matrix multiplication tensor has entries T[i,j,k] = 1 for all valid indices.
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For the standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    n, m, p = n, m, p
    # Build the tensor
    T = np.ones((n, m, p))
    # Check decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i * m + j, :], V[j * p + k, :]), W[k * n + i, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _hooi_decompose(n, m, p, rank, max_iter=100, tol=1e-12, seed=42):
    """
    Try to find a rank-r decomposition of the matrix multiplication tensor
    using HOOI (Higher-Order Orthogonal Iteration).
    
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    We want T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    This is a standard CP decomposition of the tensor T.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize factors randomly
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize columns
    for c in range(rank):
        norm_u = np.linalg.norm(U[:, c])
        norm_v = np.linalg.norm(V[:, c])
        norm_w = np.linalg.norm(W[:, c])
        if norm_u > 0 and norm_v > 0 and norm_w > 0:
            U[:, c] /= norm_u
            V[:, c] /= norm_v
            W[:, c] /= norm_w
    
    best_error = np.inf
    best_factors = (U.copy(), V.copy(), W.copy())
    
    for iteration in range(max_iter):
        # Update U
        for c in range(rank):
            # Residual for component c
            # R[i*m+j] = T[i,j,k] - sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
            # But T[i,j,k] = 1, so:
            # R[i,j,k] = 1 - sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
            # We want to minimize ||R - U_c * (V_c outer W_c)||_F^2
            # where (V_c outer W_c)[j*p+k, i*m+j] ... this is getting complex.
            
            # Let's use the standard HOOI approach.
            # The tensor T can be unfolded.
            # U-mode unfolding: T_U[i*m+j, j*p+k, k*n+i] ... no that's not right.
            
            # Let me think differently. The tensor T has indices (i, j, k).
            # U is indexed by (i, j) -> i*m+j
            # V is indexed by (j, k) -> j*p+k  
            # W is indexed by (k, i) -> k*n+i
            
            # This is not a standard tensor decomposition because the indices are coupled.
            # Let me reformulate.
            
            # Actually, let's just use gradient descent on the full problem.
            pass
        
        # Let's use a simpler approach: alternating least squares
        # For fixed V and W, find best U:
        # For each (i, j), we want U[i*m+j, c] such that
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is a linear system for each (i,j)
        
        # Update U
        U_new = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                # For fixed (i,j), we have variables U[i*m+j, 0..rank-1]
                # Constraints: for each k in [0,p):
                #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is p equations in rank unknowns.
                # If p <= rank, we can solve.
                if p <= rank:
                    # Build system: A @ u = b
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    # Least squares
                    try:
                        u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i * m + j, :] = u
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
                else:
                    U_new[i * m + j, :] = U[i * m + j, :]
        U = U_new
        
        # Update V
        V_new = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                # For fixed (j,k), variables V[j*p+k, 0..rank-1]
                # Constraints: for each i in [0,n):
                #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                if n <= rank:
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j * p + k, :] = v
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
                else:
                    V_new[j * p + k, :] = V[j * p + k, :]
        V = V_new
        
        # Update W
        W_new = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                # For fixed (k,i), variables W[k*n+i, 0..rank-1]
                # Constraints: for each j in [0,m):
                #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                if m <= rank:
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k * n + i, :] = w
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
                else:
                    W_new[k * n + i, :] = W[k * n + i, :]
        W = W_new
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i * m + j, :], V[j * p + k, :]), W[k * n + i, :])
                    error += (val - 1.0) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if error < tol:
            break
    
    return best_factors, best_error

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds margin
    
    # First, try to find a decomposition with rank less than n*m*p
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But we can potentially do better.
    
    # Start with the schoolbook decomposition as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try various ranks from low to high
    # Minimum possible rank is at least max(n*m, m*p, p*n) = 25
    min_rank = max(n * m, m * p, p * n)
    
    for rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            
            (U, V, W), err = _hooi_decompose(n, m, p, rank, max_iter=50, tol=1e-12, seed=seed)
            
            if err < 1e-10:
                # Verify exactly
                if _verify(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            # Check if we can go lower
            continue
    
    # Convert to lists for the output
    U_list = [[best_U[i, c] for c in range(best_rank)] for i in range(n * m)]
    V_list = [[best_V[i, c] for c in range(best_rank)] for i in range(m * p)]
    W_list = [[best_W[i, c] for c in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

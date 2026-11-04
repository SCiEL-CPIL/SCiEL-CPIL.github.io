"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid rank-r decomposition of the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] should equal W[k*n+i, r]... 
    Actually, the decomposition means:
    For matrix multiplication C = A * B where A is n×m, B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    Wait, let me re-read the convention: W[k*n+i, r]
    
    Standard tensor decomposition for matmul:
    T_{i,j,k} (where T is the multiplication tensor) has T_{i,j,k} = 1 if the indices match the product, else 0.
    Actually T_{i,j,k} corresponds to the coefficient in C[i,k] = sum_j A[i,j] B[j,k]
    
    The decomposition is: T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    
    But the interface says:
    U: (n*m, rank) -> U[i*m + j, r]
    V: (m*p, rank) -> V[j*p + k, r]  
    W: (p*n, rank) -> W[k*n + i, r]
    
    So the decomposition should satisfy:
    For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} ? No...
    
    Actually for matrix multiplication, the tensor is:
    T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    
    The standard form is: C[i,k] = sum_j A[i,j] * B[j,k]
    This is a trilinear form. The tensor T has T_{i,j,k} = 1 for all valid i,j,k (it's the identity in the middle).
    
    Wait, I think the decomposition is:
    The multiplication map can be written as:
    sum_r (sum_i U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) ... no that's not right either.
    
    Let me think again. The tensor decomposition for matmul n×m * m×p:
    We want: C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as a sum of rank-1 terms:
    C[i,k] = sum_r (sum_j u_{i,j}^{(r)} A[i,j]) * (sum_j v_{j,k}^{(r)} B[j,k]) ... no.
    
    Actually the standard tensor decomposition is:
    T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)}
    where T_{i,j,k} = 1 for all i,j,k (the multiplication tensor is all ones in the appropriate sense).
    
    Hmm, let me just verify numerically.
    """
    r = U.shape[1]
    # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # (since the matmul tensor T_{i,j,k} = 1 for all valid indices in the sense that 
    #  C[i,k] = sum_j A[i,j]B[j,k] means each (i,j,k) triple contributes with coefficient 1)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(r):
                    s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_r(n, m, p, r, time_limit=10):
    """
    Try to find a rank-r decomposition. Returns (U, V, W) or None.
    Uses a structured approach based on symmetry.
    """
    start = time.time()
    
    # For the case n=3, m=3, p=7, we exploit that n=p=3 partially
    # The tensor T_{i,j,k} for i in [0,3), j in [0,3), k in [0,7)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Try to find a decomposition where the structure is regular
    # Use the fact that for the "square" part (i,k both 0..2), we might use Strassen-like ideas
    
    # Let's try a parameterized family:
    # Assume U and V have a block structure based on j
    
    # Actually, let's just do a direct numerical search with constraints
    # Initialize with random values and use gradient descent
    
    np.random.seed(42)
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    # Use gradient descent to minimize sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(5000):
        if time.time() - start > time_limit:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    err = s - 1.0
                    loss += err * err
                    for r_idx in range(r):
                        grad_U[i*m+j, r_idx] += 2 * err * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        grad_V[j*p+k, r_idx] += 2 * err * U[i*m+j, r_idx] * W[k*n+i, r_idx]
                        grad_W[k*n+i, r_idx] += 2 * err * U[i*m+j, r_idx] * V[j*p+k, r_idx]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < 1e-10:
            break
    
    # Check if we found a valid solution
    if best_loss < 1e-6:
        # Try to round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded
        
        # Try integers
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        
        if _verify_decomposition(U_int, V_int, W_int, n, m, p):
            return U_int, V_int, W_int
    
    return None

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    Uses a time-bounded search strategy.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin from 120s limit
    
    # Initialize with schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Start from a reasonable lower bound and go up
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But for exact integer/half-integer, we might need more
    
    # Try ranks from 30 to 62 (below schoolbook)
    for r in range(30, min(63, rank_schoolbook)):
        if time.time() > deadline - 30:  # Leave 30s buffer
            break
        
        result = _try_rank_r(n, m, p, r, time_limit=8)
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = r
            best_U, best_V, best_W = U_found, V_found, W_found
            break  # Found a valid one, stop searching for lower
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

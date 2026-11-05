"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] for C[i,k] = sum_j A[i,j]*B[j,k].
    We store it as a vector of length n*m*p indexed by (i*m+j)*p + k, but actually
    the tensor indices are (i, j, k) -> contributes to C[i,k].
    For the decomposition: C[i,k] = sum_r U[i*m+j?] ... 
    Actually, let's use the standard tensor form:
    T is a 3-way tensor with dimensions (n, m, p) where T[i,j,k] = 1 if the (i,j) of A 
    and (j,k) of B multiply to contribute to C[i,k]. But that's not quite right.
    
    The matrix multiplication tensor T_{i,j,k} is defined such that:
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    In tensor decomposition form, we want:
    T[i,j,k] = sum_r U[i, r] * V[j, r] * W[k, r]
    where T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor.
    
    Wait, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    But that's not correct either. Let me think again.
    
    The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    The tensor for this bilinear map has entries T[i, j, k] which is 1 if the product A[i,j]*B[j,k]
    contributes to C[i,k]. But since each (i,j,k) triple uniquely determines a contribution,
    T[i,j,k] = 1 for all i,j,k.
    
    So we have a tensor of shape (n, m, p) with all ones, and we want to decompose it as:
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    But wait, the interface says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    This is a different indexing. Let me re-read the task.
    
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So the decomposition is:
    For the matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor decomposition uses:
    U of shape (n*m, rank): indexed by (i, j) -> row i*m + j
    V of shape (m*p, rank): indexed by (j, k) -> row j*p + k  
    W of shape (p*n, rank): indexed by (k, i) -> row k*n + i (trace convention)
    
    And the decomposition is:
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 for all valid (i, j, k)
    
    So the target tensor T has shape (n, m, p) with T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    """
    # Return a flat vector of the tensor values
    # Tensor shape: (n, m, p), all ones
    T = np.ones((n, m, p), dtype=float)
    return T

def _tensor_to_matrix(T, n, m, p):
    """Convert tensor T[i,j,k] to a matrix for least_squares.
    We'll flatten the tensor and create a linear system.
    For each (i,j,k), we have: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    """
    return T

def _compute_residuals(params, n, m, p, rank):
    """Compute residuals for the tensor decomposition.
    params: flattened array of U, V, W concatenated
    """
    n, m, p = n, m, p
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U = params[:size_U].reshape(n * m, rank)
    V = params[size_U:size_U + size_V].reshape(m * p, rank)
    W = params[size_U + size_V:].reshape(p * n, rank)
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                residuals.append(val - 1.0)
    return np.array(residuals)

def _compute_residuals_fast(params, n, m, p, rank):
    """Faster residual computation using vectorization."""
    size_U = n * m * rank
    size_V = m * p * rank
    
    U = params[:size_U].reshape(n * m, rank)
    V = params[size_U:size_U + size_V].reshape(m * p, rank)
    W = params[size_U + size_V:].reshape(p * n, rank)
    
    # For each (i, j, k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can vectorize over i, j, k
    residuals = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        # U[:, r] has shape (n*m,), V[:, r] has shape (m*p,), W[:, r] has shape (p*n,)
        # Reshape to (n, m), (m, p), (p, n)
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        
        # For each (i, j, k): Ur[i,j] * Vr[j,k] * Wr[k,i]
        # This is like an element-wise product with index coupling
        # We can compute this as: for each i, j: Ur[i,j] * (Vr[j,:] * Wr[:,i].T)
        # Or use einsum
        # Let's use a loop over i for now, it's small
        for i in range(n):
            # Ur[i, :] has shape (m,)
            # For each j: Ur[i,j] * Vr[j, :] * Wr[:, i]
            # Vr[:, i] is not quite right. Wr has shape (p, n), so Wr[:, i] has shape (p,)
            # Vr[j, :] has shape (p,)
            # So for fixed i: sum over j of Ur[i,j] * Vr[j,:] * Wr[:,i]
            # This is: (Ur[i, :] @ Vr) * Wr[:, i]
            # Ur[i, :] @ Vr: shape (p,)
            temp = Ur[i, :] @ Vr  # shape (p,)
            residuals[i, :, :] += temp[None, :] * Wr[:, i][None, :]
    
    return (residuals - 1.0).flatten()

def _als_step(params, n, m, p, rank, max_iter=100, tol=1e-12):
    """ALS refinement: alternate fixing two factors and solving for the third."""
    size_U = n * m * rank
    size_V = m * p * rank
    
    U = params[:size_U].reshape(n * m, rank).copy()
    V = params[size_U:size_U + size_V].reshape(m * p, rank).copy()
    W = params[size_U + size_V:].reshape(p * n, rank).copy()
    
    for it in range(max_iter):
        # Solve for U given V, W
        # For each (i,j): U[i*m+j, r] should satisfy sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        # Let M_U[i*m+j] be a matrix where M_U[i*m+j][k, r] = V[j*p+k, r] * W[k*n+i, r]
        # Then we want: U[i*m+j, :] @ M_U[i*m+j].T = 1 for all k
        # i.e., M_U[i*m+j] @ U[i*m+j, :].T = 1
        
        for i in range(n):
            for j in range(m):
                # Build matrix A of shape (p, rank)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                # Solve A @ x = 1
                try:
                    x, *_ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m + j, :] = x
                except:
                    pass
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                try:
                    x, *_ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p + k, :] = x
                except:
                    pass
        
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                try:
                    x, *_ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n + i, :] = x
                except:
                    pass
        
        # Check convergence
        res = _compute_residuals_fast(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), n, m, p, rank)
        if np.max(np.abs(res)) < tol:
            break
    
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _try_rank(n, m, p, rank, deadline, rng):
    """Try to find a rank-rank decomposition using random restarts + ALS."""
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    best_params = None
    best_residual = np.inf
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        params = rng.normal(0, 1, size_U + size_V + size_W)
        
        # ALS refinement
        params = _als_step(params, n, m, p, rank, max_iter=50, tol=1e-10)
        
        # Check residual
        res = _compute_residuals_fast(params, n, m, p, rank)
        max_res = np.max(np.abs(res))
        
        if max_res < best_residual:
            best_residual = max_res
            best_params = params.copy()
        
        if max_res < 1e-9:
            break
    
    if best_params is not None and best_residual < 1e-8:
        return best_params, best_residual
    return None, best_residual

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    rng = np.random.RandomState(42)
    
    # Start from rank 16 and go up
    # Schoolbook is rank 24
    for rank in range(16, 25):
        if time.time() > deadline:
            break
        
        params, residual = _try_rank(n, m, p, rank, deadline, rng)
        
        if params is not None and residual < 1e-8:
            # Found a valid decomposition
            size_U = n * m * rank
            size_V = m * p * rank
            
            U = params[:size_U].reshape(n * m, rank)
            V = params[size_U:size_U + size_V].reshape(m * p, rank)
            W = params[size_U + size_V:].reshape(p * n, rank)
            
            # Convert to list of lists with rounded values (integer or half-integer)
            # Check if values are close to half-integers
            def to_list(mat):
                mat_rounded = np.round(mat * 2) / 2
                return mat_rounded.tolist()
            
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback: schoolbook decomposition
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

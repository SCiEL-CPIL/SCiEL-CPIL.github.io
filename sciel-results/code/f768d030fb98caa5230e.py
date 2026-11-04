"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
N = n * m  # 16
M = m * p  # 20
P = p * n  # 20
SCHOOLBOOK_RANK = n * m * p  # 80

# The tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for matmul)
# Actually, the matrix multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A
# and (j,k) entry of B contribute to (i,k) entry of C.
# In the standard decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
# The tensor T has entries T[i,j,k] = 1 for all valid i,j,k.
# 
# We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
# 
# Index conventions:
# U: shape (n*m, rank) -> U[i*m+j, r]
# V: shape (m*p, rank) -> V[j*p+k, r]  
# W: shape (p*n, rank) -> W[k*n+i, r]  (trace convention: W[k*n+i, r])
#
# So: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

def build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k."""
    T = np.zeros((n, m, p))
    T[:] = 1.0
    return T

def check_decomposition(U, V, W, tol=1e-6):
    """Check if the decomposition exactly reconstructs the tensor."""
    T = build_tensor()
    rank = U.shape[1]
    # Reshape for efficient computation
    U_r = U.T  # (rank, N)
    V_r = V.T  # (rank, M)
    W_r = W.T  # (rank, P)
    
    # For each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Efficient way: for each r, form outer product of U_r[r], V_r[r], W_r[r]
    # and sum over r
    result = np.zeros((n, m, p))
    for r in range(rank):
        u_vec = U_r[r].reshape(n, m)
        v_vec = V_r[r].reshape(m, p)
        w_vec = W_r[r].reshape(p, n)
        # The contribution to T[i,j,k] from rank r is u_vec[i,j]*v_vec[j,k]*w_vec[k,i]
        # This is a bit tricky to vectorize. Let's do it element-wise.
        # For each (i,j,k): u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
        # We can compute this as: for each j, u[:,j] (n,) * v[j,:] (p,) gives (n,p)
        # then multiply by w[:, j]... no, w is indexed by (k,i)
        # Let's just loop over j
        for j in range(m):
            # u_col = u_vec[:, j] shape (n,)
            # v_row = v_vec[j, :] shape (p,)
            # For each (i,k): u_col[i] * v_row[k] * w_vec[k, i]
            # w_vec is (p, n), so w_vec[k, i]
            contrib = np.outer(u_vec[:, j], v_vec[j, :])  # (n, p)
            # Multiply element-wise by w_vec.T which is (n, p) with w_vec.T[i,k] = w_vec[k,i]
            contrib *= w_vec.T  # (n, p)
            result[:, j, :] += contrib
    return np.allclose(result, T, atol=tol, rtol=0)

def check_decomposition_fast(U, V, W, tol=1e-6):
    """Faster check using matrix operations."""
    T = build_tensor()
    rank = U.shape[1]
    U_r = U.T  # (rank, N)
    V_r = V.T  # (rank, M)
    W_r = W.T  # (rank, P)
    
    result = np.zeros((n, m, p))
    for r in range(rank):
        u_vec = U_r[r].reshape(n, m)
        v_vec = V_r[r].reshape(m, p)
        w_vec = W_r[r].reshape(p, n)
        for j in range(m):
            contrib = np.outer(u_vec[:, j], v_vec[j, :]) * w_vec.T
            result[:, j, :] += contrib
    return np.allclose(result, T, atol=tol, rtol=0)

def schoolbook():
    """Return the schoolbook decomposition."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def random_decomposition(rank, seed=42):
    """Generate a random decomposition."""
    rng = np.random.RandomState(seed)
    U = rng.randn(N, rank)
    V = rng.randn(M, rank)
    W = rng.randn(P, rank)
    return U, V, W

def objective(params, U_shape, V_shape, W_shape):
    """Objective function for optimization: minimize ||T - sum_r u_r v_r w_r||^2."""
    N = U_shape[0]
    M = V_shape[0]
    P = W_shape[0]
    rank = U_shape[1]
    
    U = params[:N*rank].reshape(U_shape)
    V = params[N*rank:N*rank+M*rank].reshape(V_shape)
    W = params[N*rank+M*rank:].reshape(W_shape)
    
    T = build_tensor()
    result = np.zeros((n, m, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        for j in range(m):
            result[:, j, :] += np.outer(u_vec[:, j], v_vec[j, :]) * w_vec.T
    
    diff = result - T
    return np.sum(diff * diff)

def try_optimize(rank, max_iters=1000, time_limit=10):
    """Try to find a decomposition of given rank via optimization."""
    start_time = time.time()
    T = build_tensor()
    
    # Initialize with random values
    rng = np.random.RandomState(0)
    U = rng.randn(N, rank)
    V = rng.randn(M, rank)
    W = rng.randn(P, rank)
    
    # Scale to help optimization
    scale = (n * m * p / (rank * 3)) ** (1/3)
    U *= scale
    V *= scale
    W *= scale
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_loss = np.inf
    best_params = params.copy()
    
    for attempt in range(3):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb initial guess
        if attempt > 0:
            params = best_params + rng.randn(len(params)) * 0.1
        
        # Use L-BFGS-B
        res = minimize(objective, params, args=(U.shape, V.shape, W.shape), 
                       method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': 1e-10})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x.copy()
    
    if time.time() - start_time > time_limit:
        return None, best_loss
    
    # Check if we found an exact solution
    U_opt = best_params[:N*rank].reshape(N, rank)
    V_opt = best_params[N*rank:N*rank+M*rank].reshape(M, rank)
    W_opt = best_params[N*rank+M*rank:].reshape(P, rank)
    
    if check_decomposition_fast(U_opt, V_opt, W_opt, tol=1e-6):
        return (U_opt, V_opt, W_opt), best_loss
    
    return None, best_loss

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = SCHOOLBOOK_RANK
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Known bounds: for 4x4x5, the rank is at most 80 (schoolbook)
    # Lower bounds: at least max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # Also at least n+p = 9, etc.
    # Realistic target: try ranks from 20 upward
    
    # First, let's try a few random restarts at various ranks
    # We'll search from low to high rank
    
    for rank in range(20, SCHOOLBOOK_RANK + 1):
        if time.time() - start_time > deadline - 30:  # Leave 30s buffer
            break
        
        # Try optimization for this rank
        result, loss = try_optimize(rank, max_iters=500, time_limit=15)
        
        if result is not None:
            U_opt, V_opt, W_opt = result
            # Verify
            if check_decomposition_fast(U_opt, V_opt, W_opt, tol=1e-8):
                best_rank = rank
                best_result = (U_opt, V_opt, W_opt)
                break  # Found minimal rank, stop
    
    # If we didn't find anything better, return schoolbook
    U_best, V_best, W_best = best_result
    
    # Convert to lists as required
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

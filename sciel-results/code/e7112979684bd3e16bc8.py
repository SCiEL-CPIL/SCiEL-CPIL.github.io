"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]
import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the tensor."""
    T_target = _build_tensor(n, m, p)
    T_recon = np.zeros_like(T_target)
    for c in range(U.shape[1]):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return np.max(np.abs(T_recon - T_target)) < tol

def _als_optimize(U, V, W, n, m, p, max_iters=100, deadline=None):
    """ALS optimization for tensor decomposition."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve linear system
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                M = np.zeros((p, U.shape[1]))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                t = np.ones(p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                    U_new[i*m+j, :] = sol
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        U = U_new
        
        # Update V: for each (j,k), solve linear system
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                M = np.zeros((n, V.shape[1]))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                t = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                    V_new[j*p+k, :] = sol
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        V = V_new
        
        # Update W: for each (k,i), solve linear system
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                M = np.zeros((m, W.shape[1]))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                t = np.ones(m)
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                    W_new[k*n+i, :] = sol
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        W = W_new
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual norm of the decomposition."""
    T_target = _build_tensor(n, m, p)
    T_recon = np.zeros_like(T_target)
    for c in range(U.shape[1]):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return np.linalg.norm(T_recon - T_target)

def _try_rank(n, m, p, rank, max_attempts=10, deadline=None):
    """Try to find a valid decomposition at the given rank."""
    best_residual = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt % 3 == 0:
            # Random initialization
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
        elif attempt % 3 == 1:
            # Sparse initialization
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                # Randomly select one entry in each
                idx_u = np.random.randint(0, n*m)
                idx_v = np.random.randint(0, m*p)
                idx_w = np.random.randint(0, p*n)
                U[idx_u, c] = 1.0
                V[idx_v, c] = 1.0
                W[idx_w, c] = 1.0
        else:
            # Structured initialization based on schoolbook pattern
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
        
        # ALS optimization
        U, V, W = _als_optimize(U, V, W, n, m, p, max_iters=50, deadline=deadline)
        
        # Compute residual
        residual = _compute_residual(U, V, W, n, m, p)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check for exact solution
        if residual < 1e-9:
            return True, U, V, W
    
    return False, best_U, best_V, best_W

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Deadline: 140 seconds (10s safety margin from 150s limit)
    deadline = time.time() + 140
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound up to schoolbook_rank - 1
    # Focus on promising ranges first
    ranks_to_try = list(range(lower_bound, schoolbook_rank))
    
    # Prioritize ranks near known good values and with good structure
    # For 2x3x9, try ranks that are multiples or have special properties
    priority_ranks = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 42, 45, 48, 51, 53]
    ranks_to_try = [r for r in priority_ranks if lower_bound <= r < schoolbook_rank] + \
                   [r for r in ranks_to_try if r not in priority_ranks]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try this rank with multiple attempts
        found, U, V, W = _try_rank(n, m, p, rank, max_attempts=5, deadline=deadline)
        
        if found:
            # Convert to lists
            U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better rank found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END

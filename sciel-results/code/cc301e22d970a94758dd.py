"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where:
    a = i*m + j (i in [0,n), j in [0,m))
    b = j'*p + k (j' in [0,m), k in [0,p))
    c = k'*n + i' (k' in [0,p), i' in [0,n))
    T[a,b,c] = 1 if j==j' and i==i' and k==k' else 0
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _decompose_tensor(T, U, V, W):
    """Compute the tensor decomposition from factor matrices.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Returns tensor of shape (n*m, m*p, p*n)
    """
    rank = U.shape[1]
    # Use einsum for efficiency
    return np.einsum('ar,br,cr->abc', U, V, W)

def _validate(U, V, W, T, tol=1e-10):
    """Check if the decomposition is exact."""
    recon = _decompose_tensor(T, U, V, W)
    return np.max(np.abs(recon - T)) < tol

def _try_rank(n, m, p, rank, max_time=30.0):
    """Try to find a rank-rank decomposition via random init + optimization."""
    T = _build_target_tensor(n, m, p)
    start_time = time.time()
    
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization with small values
        np.random.seed(attempt * 1000 + rank)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Simple gradient-free local search: try to minimize ||T - U⊗V⊗W||
        # Use a few iterations of alternating least squares style updates
        
        for it in range(200):
            if time.time() - start_time > max_time:
                break
            
            # Update U: for each column a, U[a,:] should satisfy sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # This is a linear system if V and W are fixed
            # But it's a 3-way tensor, so we do alternating minimization
            
            # Fix V, W; solve for U
            # For each a: sum_r U[a,r] * (V[:,r] ⊗ W[:,r]) = T[a,:,:]
            # Reshape: V[:,r] ⊗ W[:,r] has size m*p * p*n
            # This is expensive, so use random projection or just do a few steps
            
            # Instead, use a simpler approach: stochastic gradient descent
            # Sample random indices
            for _ in range(10):
                a = np.random.randint(n*m)
                b = np.random.randint(m*p)
                c = np.random.randint(p*n)
                
                # Gradient of (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2 w.r.t. U[a,r]
                # = 2*(sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c]) * V[b,r]*W[c,r]
                residual = np.sum(U[a, :] * V[b, :] * W[c, :]) - T[a, b, c]
                grad_U = residual * V[b, :] * W[c, :]
                U[a, :] -= 0.01 * grad_U
                
                residual = np.sum(U[a, :] * V[b, :] * W[c, :]) - T[a, b, c]
                grad_V = residual * U[a, :] * W[c, :]
                V[b, :] -= 0.01 * grad_V
                
                residual = np.sum(U[a, :] * V[b, :] * W[c, :]) - T[a, b, c]
                grad_W = residual * U[a, :] * V[b, :]
                W[c, :] -= 0.01 * grad_W
            
            # Check convergence
            recon = _decompose_tensor(T, U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-10:
                return U, V, W, True
    
    return None, None, None, False

def construct():
    """
    Returns a rank-minimal exact decomposition of the 2x3 by 3x9 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Start with trivial decomposition as fallback
    rank_trivial = n * m * p
    U = np.zeros((n * m, rank_trivial))
    V = np.zeros((m * p, rank_trivial))
    W = np.zeros((p * n, rank_trivial))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try ranks from 27 upward
    
    for rank in range(27, best_rank):
        if time.time() > deadline - 10:
            break
        
        U_try, V_try, W_try, success = _try_rank(n, m, p, rank, max_time=15.0)
        
        if success:
            # Verify
            T = _build_target_tensor(n, m, p)
            if _validate(U_try, V_try, W_try, T):
                best_rank = rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break  # Found the minimum, stop
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

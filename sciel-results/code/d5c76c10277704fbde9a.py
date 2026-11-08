"""Matrix multiplication tensor decomposition for 2x5 x 5x6."""
DIMENSIONS = [2, 5, 6]
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    """Check whether U,V,W exactly decompose the matrix multiplication tensor.
    Returns True if for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1.
    Uses integer-ish tolerance since entries should be integer/half-integer.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification using einsum."""
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    # For each (i,j,k), sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r]
    # This is: sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # First: A[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
    A = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
    # Then: T[i,k] = sum_r A[i,k,r] * W[k,i,r]
    # Need to index W[k,i,r] - transpose W to (k,n,r) which it already is
    T = np.einsum('ikr,kir->ik', A, W_reshaped)
    # T should be all ones, shape (n, p)
    return np.allclose(T, np.ones((n, p)), atol=1e-9)


def _try_rank_decomposition(n, m, p, rank, time_budget=30.0, seed=42):
    """Attempt to find a rank-`rank` decomposition using random initialization + ALS.
    Returns (U, V, W) if successful, else None.
    """
    if rank < n * m * p:
        # Check if rank is at least the minimum possible
        # Lower bound: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
        # Also, for matrix mult (n x m) * (m x p), rank >= n*m*p / min(n,m,p) roughly
        pass

    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to get reasonable magnitudes
    U *= 1.0 / np.sqrt(n * m)
    V *= 1.0 / np.sqrt(m * p)
    W *= 1.0 / np.sqrt(p * n)
    
    start = time.time()
    max_iter = 500
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # ALS step: fix V, W, solve for U
        # For each (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r] summed over k) = 1 for all k
        # This is complex. Let's use gradient-based approach instead.
        
        # Compute residuals
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # A[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
        A = np.einsum('ijr,jkr->ikr', U_r, V_r)
        # T[i,k] = sum_r A[i,k,r] * W[k,i,r]
        T = np.einsum('ikr,kir->ik', A, W_r)
        # Residual: T - ones
        resid = T - np.ones((n, p))
        loss = np.sum(resid ** 2)
        
        if loss < 1e-12:
            # Found a solution, verify
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
        
        # Gradient descent step (simple)
        lr = 0.01 / (1.0 + 0.01 * it)
        
        # Gradient w.r.t. W[k,n,r]:
        # dT[i,k]/dW[k,n,r] = A[i,k,r]  (for the specific i=n' that matches)
        # Actually T[i,k] = sum_r A[i,k,r]*W[k,i,r]
        # dT[i,k]/dW[k,i,r] = A[i,k,r]
        # So grad_W[k,n,r] = sum_i resid[i,k] * A[i,k,r]  ... wait, need to be careful
        # T[i,k] depends on W[k,i,r]
        # dT[i,k]/dW[k,i',r'] = A[i,k,r'] * delta(i',i)
        # So grad_W[k,i,r] = resid[i,k] * A[i,k,r]
        # grad_W shape: (p, n, rank)
        grad_W = np.zeros_like(W_r)
        for i in range(n):
            for k in range(p):
                grad_W[k, i, :] = resid[i, k] * A[i, k, :]
        
        # Gradient w.r.t. V[j,k,r]:
        # A[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
        # T[i,k] = sum_r A[i,k,r] * W[k,i,r]
        # dT[i,k]/dV[j,k,r] = sum_r' W[k,i,r'] * dA[i,k,r']/dV[j,k,r] * delta(r',r)
        #                  = W[k,i,r] * U[i,j,r]
        # grad_V[j,k,r] = sum_i resid[i,k] * U[i,j,r] * W[k,i,r]
        # But U[i,j,r] and W[k,i,r] need to be indexed properly.
        # grad_V[j,k,r] = sum_i resid[i,k] * U_r[i,j,r] * W_r[k,i,r]
        grad_V = np.zeros_like(V_r)
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    grad_V[j, k, :] += resid[i, k] * U_r[i, j, :] * W_r[k, i, :]
        
        # Gradient w.r.t. U[i,j,r]:
        # A[i,k,r] = sum_j' U[i,j',r] * V[j',k,r]
        # dA[i,k,r]/dU[i,j,r] = V[j,k,r]
        # dT[i,k]/dU[i,j,r] = sum_r' W[k,i,r'] * V[j,k,r'] * delta(r',r)
        #                   = W[k,i,r] * V[j,k,r]
        # grad_U[i,j,r] = sum_k resid[i,k] * V[j,k,r] * W[k,i,r]
        grad_U = np.zeros_like(U_r)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    grad_U[i, j, :] += resid[i, k] * V_r[j, k, :] * W_r[k, i, :]
        
        U_r -= lr * grad_U
        V_r -= lr * grad_V
        W_r -= lr * grad_W
        
        U = U_r.reshape(-1, rank)
        V = V_r.reshape(-1, rank)
        W = W_r.reshape(-1, rank)
    
    # Final check
    if _verify_fast(U, V, W, n, m, p):
        return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s safety margin out of 120s
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound for matrix multiplication rank:
    # For (n x m) * (m x p), rank >= n*m*p / max(n, m, p) is a weak bound
    # Actually, known results: rank of 2x5x5 matmul is 25 (schoolbook) but maybe less?
    # For small matrices, the rank is often the schoolbook rank or slightly less.
    # Let's try ranks from a lower bound up to schoolbook rank - 1
    
    # Lower bound: at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But actually, each factor has at most n*m, m*p, p*n nonzero rows respectively
    # so rank >= max(n*m, m*p, p*n) = 30
    
    lb = max(n * m, m * p, p * n)
    sb_rank = n * m * p
    
    # Try a few candidate ranks below schoolbook
    # Given time constraints, try a small number of ranks
    for rank in range(sb_rank - 1, lb - 1, -1):
        if time.time() > deadline - 20.0:
            break
        
        # Try with a few different seeds
        for seed in [42, 123, 7]:
            if time.time() > deadline - 10.0:
                break
            
            result = _try_rank_decomposition(n, m, p, rank, time_budget=15.0, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to exact fractions if possible (entries should be integer/half-integer)
                # For now, return as is if verification passes
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END

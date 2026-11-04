"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1 if output index (k*n+i) and input indices (i*m+j, j*p+k)
    # The tensor is: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    # and for all other index combinations where not (i,j,k) pattern, sum should be 0.
    # Actually the tensor is defined as: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a, b, c] = 0.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We need to check this equals the true tensor.
    
    # True tensor: T[a, b, c] = 1 iff exists i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # a ranges 0..n*m-1, b ranges 0..m*p-1, c ranges 0..p*n-1
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Reconstruct
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness
    if not np.allclose(T_recon, T_true, atol=1e-10):
        return False
    return True

def _try_rank(r, n, m, p, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition using iterative refinement."""
    if r < max(n, m, p):
        return None  # Rank must be at least max(n,m,p)
    
    rng = np.random.RandomState(seed)
    
    # Try multiple random initializations
    for init_seed in range(5):
        rng = np.random.RandomState(seed + init_seed)
        
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Build true tensor
        T_true = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m + j, j*p + k, k*n + i] = 1.0
        
        # Iterative refinement: alternate updating one factor
        for it in range(max_iters):
            # Update U: U[a, r] = sum_{b,c} T_true[a,b,c] * V[b,r] * W[c,r] / (sum_{b,c} V[b,r]^2 * W[c,r]^2 + eps)
            # Actually use ALS-style: for each a, U[a,:] = argmin sum_{b,c} (T_true[a,b,c] - U[a,:]·(V[b,:]*W[c,:]))^2
            # This is a linear system: U[a,:] * (V * W_outer) = T_true[a,:,:] weighted
            # Simpler: gradient descent approach
            
            # Compute residuals
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon  # residual
            
            # Update U: U[a, r] += eta * sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
            # Update V: V[b, r] += eta * sum_{a,c} R[a,b,c] * U[a,r] * W[c,r]
            # Update W: W[c, r] += eta * sum_{a,b} R[a,b,c] * U[a,r] * V[b,r]
            
            # Use fixed small learning rate
            eta = 0.01
            
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            # Check convergence
            if it % 100 == 0:
                err = np.max(np.abs(T_recon - T_true))
                if err < 1e-10:
                    # Found a solution, verify exactly
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        # Round to nearest integers or half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
                        # Try integer rounding
                        U_r = np.round(U)
                        V_r = np.round(V)
                        W_r = np.round(W)
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
                    return None
        
        # After iterations, try to snap to nice values
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
    
    return None

def _schoolbook(n, m, p):
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
    deadline = time.time() + 120  # 120 second budget
    
    # Try to find a lower-rank decomposition
    # Lower bound on rank: max(n*m, m*p, p*n) for matrix mult? Actually rank >= max(n,m,p)
    # But for matrix multiplication, known lower bounds are stronger.
    # For 3x4x8: schoolbook is 96. Let's try ranks below that.
    
    # Try a few candidate ranks
    candidates = [32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
    
    # Filter: rank must be at least max(n, m, p) = 8
    # Also rank must be at least the dimension of the tensor's "support"
    
    for r in candidates:
        if time.time() > deadline - 30:
            break
        if r >= n * m * p:
            break  # Schoolbook already handles this
        
        result = _try_rank(r, n, m, p, max_iters=2000, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

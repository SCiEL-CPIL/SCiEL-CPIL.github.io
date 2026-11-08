import numpy as np
from fractions import Fraction
import time
import random
import math

DIMENSIONS = [2, 4, 7]
_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # Check that each factor has correct dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # For each output element (i, k), check that the sum over j of A[i,j]*B[j,k]
    # equals sum over r of U[i*m+j_r, r] * V[j_r*p+k, r] * W[k*n+i, r]
    # More precisely: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if we're decomposing the multiplication tensor
    
    # Build the multiplication tensor
    # T[i, j, k] = 1 (the standard matrix mult tensor has 1s)
    # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute the reconstructed tensor
    # For each (i, j, k): sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    # This is a bit complex; let's do it more carefully
    
    # Actually, the standard form is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] * A[i,j] * B[j,k] summed over i,j,k gives (A*B)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right either. Let me think more carefully.
    
    # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all valid i,j,k
    # (since (AB)_{ik} = sum_j A_{ij} B_{jk})
    
    # A rank-r decomposition means:
    # M[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U is indexed by (i,j), V by (j,k), W by (k,i)
    
    # In our flat indexing:
    # U_flat[i*m + j, r] = U[i,j,r]
    # V_flat[j*p + k, r] = V[j,k,r]
    # W_flat[k*n + i, r] = W[k,i,r]
    
    # So we need: for all i, j, k:
    # sum_r U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r] == 1
    
    # Let's verify a sample of entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_rank(n, m, p, target_rank, max_iterations=500):
    """
    Try to find a rank-target_rank decomposition using a structured approach.
    Uses random initialization and gradient descent on the tensor reconstruction error.
    """
    if _time_left() < 5.0:
        return None
    
    n, m, p = n, m, p
    r = target_rank
    
    # Initialize with random values (scaled)
    np.random.seed(42)
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    # We want: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # This is a non-convex problem. Use gradient descent.
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iterations):
        if _time_left() < 10.0:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # val = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = val - 1.0
                    loss += err * err
                    
                    # d/dU[i*m+j, r] = 2*err * V[j*p+k, r] * W[k*n+i, r]
                    grad_U[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                    # d/dV[j*p+k, r] = 2*err * U[i*m+j, r] * W[k*n+i, r]
                    grad_V[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                    # d/dW[k*n+i, r] = 2*err * U[i*m+j, r] * V[j*p+k, r]
                    grad_W[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-8:
            break
    
    if best_loss < 1e-6:
        # Convert to fractions for exactness check
        # Round to nearest half-integer or integer
        U_frac = [[round(float(x) * 2) / 2 for x in row] for row in best_U]
        V_frac = [[round(float(x) * 2) / 2 for x in row] for row in best_V]
        W_frac = [[round(float(x) * 2) / 2 for x in row] for row in best_W]
        
        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def _try_bilinear_structured(n, m, p, target_rank):
    """
    Try a structured decomposition based on bilinear forms.
    For 2x4 * 4x7, we can exploit the structure.
    """
    if _time_left() < 5.0:
        return None
    
    n, m, p = n, m, p
    r = target_rank
    
    # For 2x4 * 4x7:
    # Output is 2x7. Each output (i,k) depends on sum over j=0..3 of A[i,j]*B[j,k]
    # A has 8 elements, B has 28 elements.
    
    # Try a decomposition where we group the j dimension
    # One approach: for each i (2 values) and k (7 values), we need to compute
    # sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # A simple structured approach: use r = n * p + m * p - something
    # Or try r = n * m + m * p - m (overlapping)
    
    # Let's try a Kronecker-like structure
    # For each j, we have a 2x7 subproblem (A[:,j] outer B[j,:])
    # Total rank if independent: 2 * 4 * 7 = 56
    # Can we share factors?
    
    # Try: for each i in range(n), k in range(p):
    #   (AB)[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as:
    #   = sum_j (A[i,j] * B[j,k])
    # 
    # If we define for each j: U_j is a vector, V_j is a vector, W_j is a vector
    # such that the product gives the right contribution.
    
    # Actually, let's try a different structured approach.
    # Group by pairs of (i, j) or (j, k)
    
    # For now, try the random gradient approach with different seeds
    for seed in range(10):
        if _time_left() < 5.0:
            break
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        lr = 0.05
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        max_iter = 200
        for it in range(max_iter):
            if _time_left() < 10.0:
                break
            
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = val - 1.0
                        loss += err * err
                        grad_U[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                        grad_V[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                        grad_W[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-8:
                break
        
        if best_loss < 1e-6:
            U_frac = [[round(float(x) * 2) / 2 for x in row] for row in best_U]
            V_frac = [[round(float(x) * 2) / 2 for x in row] for row in best_V]
            W_frac = [[round(float(x) * 2) / 2 for x in row] for row in best_W]
            
            if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known bounds: for 2x4 * 4x7, the rank is between some lower bound and 56
    # Lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # So rank is at least 28.
    
    # Try ranks from lower bound up to schoolbook - 1
    for r in range(28, best_rank):
        if _time_left() < 15.0:
            break
        
        result = _try_structured_rank(n, m, p, r, max_iterations=300)
        if result is not None:
            best = result
            best_rank = r
            break  # Found the minimum, stop
        
        result = _try_bilinear_structured(n, m, p, r)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
# EVOLVE-BLOCK-END

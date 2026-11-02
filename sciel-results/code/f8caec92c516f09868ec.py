"""Matrix multiplication 2x3x9 rank-minimal decomposition using Kronecker structure."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    Strategy: Exploit the structure 9 = 3*3 by viewing the 3x9 matrix as a 3x3 block matrix
    where each block is 1x3. We use the fact that the 3x9 multiplication can be decomposed
    using the Kronecker product structure.
    
    The key insight: A 2x3 matrix times a 3x9 matrix can be seen as:
    - The 3x9 matrix has 3 rows, each of length 9
    - We can decompose the 9 columns into 3 groups of 3
    
    Using the group algebra of Z_3, we can represent the 3x3 multiplication over the ring
    Z_3[x]/(x^3-1), which has rank 3 for the multiplication itself.
    
    Actually, let's use a more direct approach:
    For 2x3x9, the schoolbook rank is 54. We can potentially do better by exploiting
    the structure.
    
    Let's try to construct a decomposition of rank 27 first (the lower bound), then
    work up if needed.
    
    The multiplication tensor for (2,3,9) has T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,9).
    We need: sum_r U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r] = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Lower bound on rank: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try ranks from 27 upward
    
    for rank in range(27, 55):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        found = _try_decompose(n, m, p, rank, deadline)
        if found is not None:
            return found
    
    # Fallback: schoolbook decomposition with rank 54
    return _schoolbook(n, m, p)

def _try_decompose(n, m, p, rank, deadline):
    """Try to find a rank decomposition using various strategies."""
    # Strategy 1: Use Kronecker product structure
    # Since p = 9 = 3*3, we can try to exploit this
    
    # The 3x9 multiplication can be viewed as 3 multiplications of 3x3 blocks
    # if we reshape the 9 columns into 3 groups of 3
    
    # Let's try a structured approach using the fact that 9 = 3*3
    # We can use the decomposition of 2x3x3 (rank 6) and tensor with something
    
    # Actually, let's use a more systematic approach:
    # For 2x3x9, we can write the 3x9 matrix as [B1 | B2 | B3] where each Bj is 3x3
    # Then A * [B1|B2|B3] = [A*B1 | A*B2 | A*B3]
    # Each A*Bj is a 2x3 matrix multiplication, which has rank 6
    # So total rank would be 3*6 = 18 if we could share structure
    
    # But this doesn't directly give a lower rank because the outputs are separate.
    
    # Let's try a direct numerical approach with better initialization
    
    np.random.seed(42)
    
    # Initialize with a structured pattern
    # Use the schoolbook as a starting point and try to compress
    
    # First, let's create the schoolbook decomposition
    U_school = np.zeros((n * m, n * m * p), dtype=float)
    V_school = np.zeros((m * p, n * m * p), dtype=float)
    W_school = np.zeros((p * n, n * m * p), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Now try to find a rank-revealing decomposition
    # We can use the fact that the tensor is the sum of 54 rank-1 tensors
    # and try to combine them
    
    # Alternative: use random initialization and optimize
    
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(20):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(42 + trial * 7)
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Optimize using gradient descent with adaptive learning rate
        lr = 0.01
        best_local_residual = float('inf')
        
        for it in range(500):
            if time.time() > deadline - 15:
                break
            
            # Compute residual
            res = _compute_residual(U, V, W, n, m, p)
            residual_norm = np.max(np.abs(res))
            
            if residual_norm < 1e-10:
                return _finalize(U, V, W, n, m, p)
            
            if residual_norm < best_local_residual:
                best_local_residual = residual_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = res[i, j, k]
                        dU[i * m + j, :] += r * V[j * p + k, :] * W[k * n + i, :]
                        dV[j * p + k, :] += r * U[i * m + j, :] * W[k * n + i, :]
                        dW[k * n + i, :] += r * U[i * m + j, :] * V[j * p + k, :]
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 0:
                effective_lr = lr / (1 + grad_norm)
                U -= effective_lr * dU
                V -= effective_lr * dV
                W -= effective_lr * dW
            
            # Decay learning rate
            lr *= 0.999
        
        if best_local_residual < best_residual:
            best_residual = best_local_residual
            best_U, best_V, best_W = best_U.copy(), best_V.copy(), best_W.copy()
        
        if best_residual < 1e-8:
            return _finalize(best_U, best_V, best_W, n, m, p)
    
    return None

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor."""
    res = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[i, j, k] = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]) - 1.0
    return res

def _finalize(U, V, W, n, m, p):
    """Finalize and verify the decomposition."""
    # Verify
    res = _compute_residual(U, V, W, n, m, p)
    if np.max(np.abs(res)) < 1e-8:
        # Try to round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        res_rounded = _compute_residual(U_rounded, V_rounded, W_rounded, n, m, p)
        if np.max(np.abs(res_rounded)) < 1e-6:
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

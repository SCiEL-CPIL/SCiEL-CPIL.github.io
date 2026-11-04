"""Matrix multiplication 3,4,6 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 140.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c] should equal W[k*n+i,c]
    Actually: the tensor T[i,j,k] = 1 for all valid (i,j,k)
    Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all (i,j,k)
    """
    rank = U.shape[1]
    # Build the tensor
    T = np.ones((n, m, p))
    # Compute decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using tensor contraction."""
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can compute this as: for each c, outer product
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # This is a 3-way contraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    return np.allclose(T, 1.0, atol=1e-9)


def _verify_vec(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _refine(U, V, W, n, m, p, max_iters=200, tol=1e-10):
    """Local refinement using gradient descent on the bilinear tensor decomposition.
    Minimize ||T - sum_c U_c ⊗ V_c ⊗ W_c||^2 where T is all-ones tensor.
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank).copy()
    V3 = V.reshape(m, p, rank).copy()
    W3 = W.reshape(p, n, rank).copy()
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
    
    for it in range(max_iters):
        # Compute current tensor
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        loss = np.sum((T - 1.0)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        # dL/dU3[i,j,c] = 2*(T[i,j,k]-1) * V3[j,k,c] * W3[k,i,c] summed over k
        # dL/dV3[j,k,c] = 2*(T[i,j,k]-1) * U3[i,j,c] * W3[k,i,c] summed over i
        # dL/dW3[k,i,c] = 2*(T[i,j,k]-1) * U3[i,j,c] * V3[j,k,c] summed over j
        
        diff = T - 1.0
        
        # Gradient for U3: for each (i,j,c), sum_k diff[i,j,k] * V3[j,k,c] * W3[k,i,c]
        grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V3, W3) * 2
        grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U3, W3) * 2
        grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U3, V3) * 2
        
        # Gradient step
        U3 -= lr * grad_U
        V3 -= lr * grad_V
        W3 -= lr * grad_W
        
        # Adaptive learning rate
        new_T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        new_loss = np.sum((new_T - 1.0)**2)
        if new_loss > loss * 1.1:
            lr *= 0.5
        elif new_loss < loss * 0.9:
            lr = min(lr * 1.1, 0.1)
    
    return best_U.reshape(-1, rank), best_V.reshape(-1, rank), best_W.reshape(-1, rank)


def _try_rank(r, n, m, p, deadline, num_trials=5):
    """Try to find a valid decomposition at rank r."""
    for trial in range(num_trials):
        if time.time() > deadline:
            return None
        
        # Random initialization with small values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Try refinement
        U_r, V_r, W_r = _refine(U, V, W, n, m, p, max_iters=300, tol=1e-12)
        
        if _verify_vec(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    
    return None


def _try_structured_rank(r, n, m, p, deadline):
    """Try structured decompositions based on known results.
    For 3x4x4 matmul, rank is 18 (Winograd). For 3x4x6, we can try
    decomposing the 4x6 part using known low-rank decompositions.
    
    Actually, let's try a different approach: use the fact that
    matmul(3,4,6) can be viewed as 3 independent (4,6) multiplications
    with shared structure, or use block decomposition.
    
    Let's try rank based on known bounds:
    - Trivial: 3*4*6 = 72
    - Winograd-style: for 3x3x3, rank 23. For 3x4x4, rank 18.
    - For 3x4x6: we can split 6 into 3+3, giving 2*(3x4x3) terms
      3x4x3: at most 3*4*3=36, but Winograd gives ~18-20
    - So total maybe ~40-48
    
    Let's just do a search.
    """
    return None


def construct():
    """Find rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Start from schoolbook and try to reduce rank
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actual minimum is likely higher.
    
    # Try ranks from low to high
    # For 3x4x6, let's start from a reasonable estimate
    # The bilinear complexity of 3x4x6 matmul:
    # We can use: C = A*B where A is 3x4, B is 4x6
    # Split B into B1 (4x3) and B2 (4x3): C = A*B1 + A*B2
    # Each 3x4x3 matmul has rank at most 36, but Winograd gives ~18
    # So total rank ~36-48
    
    # Let's try a range of ranks
    best_result = None
    
    # First, let's try a structured approach
    # For 3x4x4, the optimal rank is 18 (known result)
    # For 3x4x6 = 3x4x4 + 3x4x2
    # 3x4x2: rank at most 24, Winograd-style maybe ~12-16
    # Total: ~30-34
    
    # Let's try ranks from 24 upward
    for r in range(24, 73):
        if time.time() > deadline - 30:  # Leave 30s for final verification
            break
        
        result = _try_rank(r, n, m, p, deadline - 30, num_trials=3)
        if result is not None:
            U, V, W = result
            best_result = (r, U, V, W)
            break
    
    if best_result is not None:
        r, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
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

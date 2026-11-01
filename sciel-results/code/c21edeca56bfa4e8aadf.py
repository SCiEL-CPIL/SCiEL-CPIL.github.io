import numpy as np
import time
import random
import sys

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form an exact decomposition.
    U shape: (n*m, r)
    V shape: (m*p, r)
    W shape: (p*n, r)
    """
    r = U.shape[1]
    # Reshape U, V, W to 3D tensors
    # U[i, j, k] corresponds to U_flat[i*m + j, k]
    # V[j, k, l] corresponds to V_flat[j*p + k, l]
    # W[l, i, k] corresponds to W_flat[l*n + i, k] (Note: W index is k*n+i)
    
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r) # W[k, i, r] -> W_flat[k*n + i, r]
    
    # The multiplication tensor T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # We want T[i, j, k] to be 1 if i,j,k are valid indices for A*B? 
    # No, the tensor for matrix multiplication M_{n,m,p} is defined as:
    # M[i, j, k] = 1 if we are computing C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor entries are T[i, j, k] where i is row of A/C, j is col of A/row of B, k is col of B/C.
    # T[i, j, k] = 1.
    
    # Let's construct the target tensor
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0
                
    # Compute the decomposed tensor
    # T_decomp[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # We can compute this efficiently using einsum
    # U_3d: (n, m, r)
    # V_3d: (m, p, r)
    # W_3d: (p, n, r) -> we need W[k, i, r]. 
    # Current W_3d is W.reshape(p, n, r). So W_3d[k, i, r] is correct.
    
    T_decomp = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    error = np.max(np.abs(T_decomp - T_target))
    return error < tol

def _round_and_verify(U, V, W, n, m, p):
    """
    Attempt to round the matrices to integers or half-integers and verify.
    """
    # Try rounding to nearest integer
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _verify(U_int, V_int, W_int, n, m, p):
        return U_int, V_int, W_int
    
    # Try rounding to nearest half-integer (multiply by 2, round, divide by 2)
    U_half = np.round(U * 2) / 2
    V_half = np.round(V * 2) / 2
    W_half = np.round(W * 2) / 2
    if _verify(U_half, V_half, W_half, n, m, p):
        return U_half, V_half, W_half
        
    return None

def _newton_step(U, V, W, n, m, p, lr=0.1, max_iter=100):
    """
    Perform Newton-Raphson optimization on U, V, W to minimize
    || sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] - T_target ||^2
    """
    r = U.shape[1]
    
    # Precompute target tensor
    T_target = np.ones((n, m, p))
    
    # Reshape for easier manipulation
    # U: (n, m, r)
    # V: (m, p, r)
    # W: (p, n, r) -> W[k, i, r]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Compute current tensor
        T_decomp = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        # Compute error
        E = T_decomp - T_target
        
        # Gradient for U
        # dE/du[i,j,r] = sum_{k} E[i,j,k] * V[j,k,r] * W[k,i,r]
        # grad_U[i, j, r] = sum_k E[i, j, k] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by [k, i, r]
        grad_U = np.einsum('ijk,jkr,kir->ijr', E, V, W)
        
        # Gradient for V
        # dE/dv[j,k,r] = sum_{i} E[i,j,k] * U[i,j,r] * W[k,i,r]
        grad_V = np.einsum('ijk,ijr,kir->jkr', E, U, W)
        
        # Gradient for W
        # dE/dw[k,i,r] = sum_{j} E[i,j,k] * U[i,j,r] * V[j,k,r]
        # Note: W[k, i, r]
        grad_W = np.einsum('ijk,ijr,jkr->kir', E, U, V)
        
        # Check convergence
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm < 1e-8:
            break
            
        # Update with learning rate
        # Simple gradient descent step. Newton requires Hessian which is complex.
        # We can use a heuristic step size.
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Known lower bound is 21. Schoolbook is 36.
    # We try ranks from 21 up to 35.
    for rank in range(21, 36):
        # Allow some time for each rank
        time_limit_for_rank = 5.0
        attempts = 0
        
        while time.time() - start_time < 110 and attempts < 50:
            attempts += 1
            
            # Initialize with random small values
            # Scale is important. Since entries are integers, values shouldn't be too large.
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Reshape for optimization
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            # Optimization
            U_opt, V_opt, W_opt = _newton_step(U_3d, V_3d, W_3d, n, m, p, lr=0.05, max_iter=50)
            
            # Flatten back
            U_flat = U_opt.reshape(n * m, rank)
            V_flat = V_opt.reshape(m * p, rank)
            W_flat = W_opt.reshape(p * n, rank)
            
            # Check if it's close to integers
            # The error should be small if it converged to a valid solution
            # Try rounding
            result = _round_and_verify(U_flat, V_flat, W_flat, n, m, p)
            if result is not None:
                U_r, V_r, W_r = result
                # Convert to lists
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                
            # If not integer, check if it's a valid real decomposition (unlikely to be exact)
            # But maybe we can just return the real one if error is tiny? 
            # The task asks for "integer/half-integer entries".
            # So we must round.
            
            # If we get close but not quite, maybe try a different initialization or more steps?
            # For now, just try next random init.
            
        # If we exhausted attempts for this rank, move to next rank
        if time.time() - start_time > 110:
            break
            
    # Fallback to schoolbook if no better found
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

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank decomposition for 3x4 * 4x5 matrix multiplication.
    
    The tensor T[i,j,k] = delta_{i, i'} delta_{j, j'} delta_{k, k'} where:
    - i in [0, n) = [0, 3)
    - j in [0, m) = [0, 4)  
    - k in [0, p) = [0, 5)
    
    We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Using a random search with local optimization to find low-rank solutions.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor as a matrix (matricization)
    # T is a 3D tensor of shape (n, m, p)
    # We'll work with the "flattened" form
    
    def build_target_tensor():
        """Build the 3D tensor T[i,j,k] = 1 if indices are consistent, else 0"""
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T
    
    T = build_target_tensor()
    
    # For verification: T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is the matrix multiplication tensor where (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor for matrix mult is: T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # But that's not right. Let me reconsider.
    
    # The matrix multiplication tensor T_{ijk} is defined such that:
    # (AB)_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    # So T_{ijk} = 1 if all indices are valid, and the "contraction" gives the result.
    # Actually T_{ijk} = delta_{i,i} delta_{j,j} delta_{k,k} which is just 1 for all valid i,j,k.
    # Wait, that's not right either.
    
    # Let me think again. For matrix multiplication C = A * B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i, j, k.
    # Then sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k] = C[i,k]
    # So yes, T[i,j,k] = 1 for all i, j, k.
    
    # Now we need to find a rank-r decomposition:
    # T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Note the index conventions:
    # U has shape (n*m, r), indexed by (i, j) -> row i*m + j
    # V has shape (m*p, r), indexed by (j, k) -> row j*p + k
    # W has shape (p*n, r), indexed by (k, i) -> row k*n + i
    
    # Let's try to find a low-rank solution using optimization.
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_rank = n * m * p  # Schoolbook rank as fallback
    best_U = None
    best_V = None
    best_W = None
    
    # Try various ranks from low to high
    # Theoretical lower bound for 3x4x5 is not well known, but let's try from reasonable values
    # For 3x3x3, the best known is 23 (Landsberg). For 3x4x5, we'll try from ~40 upward.
    
    import random
    import sys
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T_recon = np.zeros((n, m, p), dtype=float)
        for c in range(r):
            u = U[:, c]  # shape (n*m,)
            v = V[:, c]  # shape (m*p,)
            w = W[:, c]  # shape (p*n,)
            
            # Reshape u to (n, m), v to (m, p), w to (p, n)
            u_mat = u.reshape((n, m))
            v_mat = v.reshape((m, p))
            w_mat = w.reshape((p, n))
            
            # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        
        return np.max(np.abs(T_recon - T)) < tol
    
    def try_rank(r, num_attempts=20, max_iters=1000):
        """Try to find a rank-r decomposition using random initialization and gradient descent."""
        if r < 1:
            return None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            # Simple gradient descent
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute T_recon
                T_recon = np.zeros((n, m, p), dtype=float)
                for c in range(r):
                    u_mat = U[:, c].reshape((n, m))
                    v_mat = V[:, c].reshape((m, p))
                    w_mat = W[:, c].reshape((p, n))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_recon[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                
                # Compute error
                error = np.max(np.abs(T_recon - T))
                if error < 1e-8:
                    return U, V, W
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = T_recon[i, j, k] - T[i, j, k]
                            for c in range(r):
                                u_val = U[i*m+j, c]
                                v_val = V[j*p+k, c]
                                w_val = W[k*n+i, c]
                                dU[i*m+j, c] += err * v_val * w_val
                                dV[j*p+k, c] += err * u_val * w_val
                                dW[k*n+i, c] += err * u_val * v_val
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check final result
            if verify_decomposition(U, V, W, tol=1e-6):
                return U, V, W
        
        return None
    
    # Try ranks from low to high
    # Start from a reasonable lower bound
    # For n x m x p, the tensor rank is at least max(n*m, m*p, p*n) / something
    # Actually, a simple lower bound is max(n, m, p) but that's weak.
    # Let's try from 30 upward.
    
    for r in range(30, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, num_attempts=10, max_iters=500)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # If we found a solution, convert to lists
    if best_U is not None:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        
        # Try to round to half-integers if possible
        # Check if rounding to nearest 0.5 still gives a valid decomposition
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-4):
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
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

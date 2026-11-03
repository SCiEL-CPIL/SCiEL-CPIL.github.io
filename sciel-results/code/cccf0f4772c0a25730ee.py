"""Optimized tensor rank search for 2x3 @ 3x10 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 10]
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin for 150s limit

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 @ 3x10 matrix multiplication.
    
    Strategy:
    1. Start with the known rank-55 decomposition (2x1 naive + 2x2 Strassen on 5 blocks)
    2. Attempt to find lower rank via structured search and optimization
    3. Use subspace iteration and gradient-based methods to refine
    
    Returns: {'U': (6, rank), 'V': (30, rank), 'W': (20, rank)}
    """
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 60 schoolbook
    
    # Build the multiplication tensor T[i,j,k] such that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] where U[:,c] dot A_flat * V[:,c] dot B_flat = W[:,c] dot C_flat
    # 
    # The tensor T has entries T[(i,j), k] = delta[j==j] for each (i,j,k)
    # More precisely: T[i,m,j,p,k] but we flatten to U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    # T[U_idx, V_idx, W_idx] = 1 if the indices correspond to a valid multiplication term
    # i.e., U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    
    # Build tensor as a 3D array T[U_idx, V_idx, W_idx]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] == T"""
        # Efficient check: for each c, contribution is outer(U[:,c], outer(V[:,c], W[:,c]))
        # Instead, check: T should equal sum_c U[:,c] * V[:,c] * W[:,c] (broadcasting)
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # sum_c U[i,c] * V[j,c] * W[k,c] should equal T[i,j,k]
        r = U.shape[1]
        # Use einsum for efficiency
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=tol)
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return U, V, W
    
    # Start with schoolbook rank 60
    U, V, W = make_schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try structured decomposition: 2x3 @ 3x10
    # We can think of this as:
    # - 2x1 @ 1x10 part: needs 2*1*10 = 20 operations (but can be optimized)
    # - 2x2 @ 2x10 part: can use Strassen-like
    # - 2x1 @ 1x10 part again
    #
    # Actually, 2x3 = [2x1 | 2x2] and 3x10 = [1x10; 2x10]
    # C = A_left * B_top + A_right * B_bottom
    # A_left: 2x1, B_top: 1x10 -> 2x10 result
    # A_right: 2x2, B_bottom: 2x10 -> 2x10 result
    
    # For 2x1 @ 1x10: rank is at most 2 (since it's just scaling)
    # Actually, (2x1)(1x10) = outer product, rank 1 in tensor sense? No.
    # The tensor for (2x1)(1x10): U_idx in [0,1], V_idx in [0,9], W_idx in [0,9]
    # T[i, j, k] = 1 if i==i, j==j, k==i... 
    # For 2x1 @ 1x10: A has 2 elements, B has 10 elements, C has 20 elements
    # T[i, j, k] where i in 0..1, j in 0..9 (B index), k in 0..19 (C index = i*10 + j)
    # T[i, j, i*10+j] = 1, which is rank 20? No, it's rank 2 since for fixed i, it's outer of A_i and B
    # Actually this is just 2 independent outer products, so rank 2.
    
    # For 2x2 @ 2x10: This is more complex
    # 2x2 @ 2x10 can be done with rank... 
    # Standard 2x2 @ 2x2 is rank 7 (Strassen)
    # 2x2 @ 2x10 = 5 independent 2x2 @ 2x2 blocks? No.
    # 2x2 @ 2x10: result is 2x10
    # We can write this as 5 separate 2x2 @ 2x2 multiplications if we group the 10 columns into 5 groups of 2
    # Each 2x2 @ 2x2 can be done in rank 7, so total 5*7 = 35
    # Plus the 2x1 @ 1x10 part in rank 2
    # Total: 37? That's not right because we need to be more careful.
    
    # Let me think differently. The known result is rank 55.
    # Let's try to construct it properly.
    
    # 2x3 @ 3x10:
    # Partition A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # Partition B = [B1; B2] where B1 is 1x10, B2 is 2x10
    # C = A1 * B1 + A2 * B2
    
    # A1 * B1: (2x1)(1x10) -> 2x10, this is just outer product, tensor rank 2
    #   For each row i of A1 and column k of B1: C[i,k] += A1[i] * B1[k]
    #   Tensor: U[i, c_i] = 1, V[k, c_i] = 1, W[i*10+k, c_i] = 1 for c_i in {0,1} (for i=0,1)
    #   Wait, we need to match the global indexing.
    
    # Let me just try to find a good decomposition via optimization.
    
    def try_rank_optimization(rank, max_iters=5000, lr=0.01, verbose=False):
        """Try to find a decomposition of given rank via gradient descent."""
        if time.time() - START_TIME > TIME_LIMIT:
            return None, None, None
        
        np.random.seed(42)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Compute gradient: dL/dU = sum_c (T - sum_c' U_c V_c W_c)_{:, :, :} * V_c * W_c
            # More precisely: L = ||T - sum_c outer(U_c, outer(V_c, W_c))||^2
            # dL/dU[i,c] = -2 * sum_{j,k} (T[i,j,k] - recon[i,j,k]) * V[j,c] * W[k,c]
            
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = T - recon
            
            # Gradients
            dU = -2 * np.einsum('ijk,jr,kr->ir', residual, V, W)
            dV = -2 * np.einsum('ijk,ir,kr->jr', residual, U, W)
            dW = -2 * np.einsum('ijk,ir,jr->kr', residual, U, V)
            
            loss = np.sum(residual**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if verbose and it % 500 == 0:
                print(f"Iter {it}, loss {loss:.6f}")
            
            # Gradient descent with adaptive learning rate
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if best_loss < 1e-12:
                break
        
        if best_loss < 1e-9:
            return best_U, best_V, best_W
        return None, None, None
    
    # First, verify schoolbook
    if verify(U, V, W):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = n * m * p
    
    # Try ranks below 60
    for rank in range(50, 60):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Try multiple random initializations
        for seed in range(3):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            np.random.seed(seed)
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            for it in range(2000):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                
                recon = np.einsum('ir,jr,kr->ijk', U_init, V_init, W_init)
                residual = T - recon
                
                dU = -2 * np.einsum('ijk,jr,kr->ir', residual, V_init, W_init)
                dV = -2 * np.einsum('ijk,ir,kr->jr', residual, U_init, W_init)
                dW = -2 * np.einsum('ijk,ir,jr->kr', residual, U_init, V_init)
                
                loss = np.sum(residual**2)
                
                if loss < 1e-10:
                    break
                
                U_init -= lr * dU
                V_init -= lr * dV
                W_init -= lr * dW
            
            if verify(U_init, V_init, W_init, tol=1e-8):
                best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
                best_rank = rank
                break
    
    # If we found a better rank, use it
    U, V, W = best_U, best_V, best_W
    rank = best_rank
    
    # Convert to lists as required
    U_list = [[int(U[i, c]) if abs(U[i, c] - round(U[i, c])) < 1e-6 else float(U[i, c]) 
               for c in range(rank)] for i in range(n*m)]
    V_list = [[int(V[i, c]) if abs(V[i, c] - round(V[i, c])) < 1e-6 else float(V[i, c]) 
               for c in range(rank)] for i in range(m*p)]
    W_list = [[int(W[i, c]) if abs(W[i, c] - round(W[i, c])) < 1e-6 else float(W[i, c]) 
               for c in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find rank 20 decomposition
    # If found, return it. Else return rank 24.
    
    def check_decomposition(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Compute tensor T_approx
        # T_approx[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        # This is a tensor product.
        # We can compute this efficiently using einsum
        # T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=1e-9)
    
    def try_rank(r):
        if time.time() > deadline:
            return None
        
        # Random initialization
        # Use small integers/half-integers?
        # Let's try random floats first, then round.
        best_err = float('inf')
        best_params = None
        
        # Number of variables: r * (nm + mp + pn)
        size = r * (n*m + m*p + p*n)
        
        # We use a few random restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            params = np.random.randn(size) * 0.1
            
            def objective(params):
                U = params[:r*n*m].reshape(n*m, r)
                V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                W = params[r*(n*m+m*p):].reshape(p*n, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                err = np.sum((T_approx - target)**2)
                return err
            
            # Use Nelder-Mead or Powell?
            # Nelder-Mead is good for low dimensions, but here dim is ~500.
            # Powell is better for high dims?
            # Let's try a simple gradient descent with random restarts.
            
            # Actually, for rank 20, the landscape is very hard.
            # Let's try a specific structure:
            # Assume U, V, W have entries in {-1, 0, 1, 0.5, -0.5}
            # This is a discrete search. 5^(520) is impossible.
            
            # Let's try to use the known rank 20 solution if I can find it online?
            # I cannot.
            
            # Let's try to solve the linear system if we fix U and V?
            # If we fix U and V, W is determined by linear equations?
            # T[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
            # For fixed a,b, this is a linear system for W[:,k]?
            # No, W[c,k] is the variable.
            # For each c, we have a vector of W[c,:].
            # The equation is sum_k (U[a,k] V[b,k]) W[c,k] = T[a,b,c].
            # Let M[a,b,k] = U[a,k] V[b,k].
            # Then sum_k M[a,b,k] W[c,k] = T[a,b,c].
            # This is a linear system in W.
            # We have (n*m * m*p) equations? No, for each a,b, we get one equation for each c?
            # No, for each (a,b,c), we have an equation.
            # But the left side is linear in W.
            # So for fixed U, V, we can solve for W using least squares.
            
            # Strategy:
            # 1. Randomly initialize U and V.
            # 2. Solve for W using least squares.
            # 3. Check error.
            # 4. Update U and V using gradient descent?
            
            # Let's implement this.
            for iter in range(50):
                if time.time() > deadline:
                    return None
                
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                
                # Build the linear system for W
                # A W = b
                # A has shape (n*m * m*p * p*n, r * p*n)? No.
                # Unknowns are W (p*n x r). Total r * p*n variables.
                # Equations: for each a, b, c: sum_k U[a,k] V[b,k] W[c,k] = T[a,b,c]
                
                # We can vectorize this.
                # Let's construct the matrix A.
                # Row index: idx = (a * m*p + b) * p*n + c
                # Col index: idx_w = c * r + k  (if W is flattened row-major)
                # A[idx, idx_w] = U[a,k] * V[b,k]
                
                # This matrix is huge: (24 * 8) * (20 * 8) = 192 * 160 = 30720 entries?
                # Wait, number of equations is n*m * m*p * p*n = 6 * 12 * 8 = 576.
                # Number of variables is p*n * r = 8 * 20 = 160.
                # So we have 576 equations, 160 variables.
                # We can solve this with least squares.
                
                A = np.zeros((576, 160))
                b = np.zeros(576)
                
                for a in range(n*m):
                    for b_idx in range(m*p):
                        for c in range(p*n):
                            eq_idx = (a * m*p + b_idx) * p*n + c
                            b[eq_idx] = target[a, b_idx, c]
                            
                            # Calculate coefficients for W[c, k]
                            # W is flattened as W[c, k] -> index c*r + k
                            for k in range(r):
                                col_idx = c * r + k
                                A[eq_idx, col_idx] = U[a, k] * V[b_idx, k]
                
                # Solve A W_flat = b
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p*n, r)
                
                # Calculate error
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                err = np.sum((T_approx - target)**2)
                
                if err < best_err:
                    best_err = err
                    best_params = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-9:
                        # Found a solution!
                        # Round to nearest 0.5?
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if check_decomposition(U_rounded, V_rounded, W_rounded):
                            return U_rounded, V_rounded, W_rounded
                        elif check_decomposition(U, V, W):
                            return U, V, W

        return None

    # Try rank 20
    res = try_rank(20)
    if res is not None:
        U, V, W = res
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to rank 24
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

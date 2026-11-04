import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = 3, 3, 4
    
    # Build the target tensor T
    # T[a, b, c] where a is A index (i*m+j), b is B index (j*p+k), c is C index (k*n+i)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    # Pre-flatten T for different modes
    # Mode 3 (W): T_flat_W[a*m*p + b, c] = T[a, b, c]
    T_flat_W = T.reshape(n*m*m*p, p*n)
    # Mode 1 (U): T_flat_U[b*p*n + c, a] = T[a, b, c]
    T_flat_U = T.transpose(1, 2, 0).reshape(m*p*p*n, n*m)
    # Mode 2 (V): T_flat_V[a*m + c, b] = T[a, b, c] ? No.
    # T[a, b, c] = sum U[a,r] V[b,r] W[c,r]
    # For V: T[a, b, c] = sum (U[a,r] W[c,r]) V[b,r]
    # Design matrix rows indexed by (a,c), cols by r.
    # Target vector for row b is T[:, b, :] flattened over (a,c).
    # T_flat_V[a*p*n + c, b] = T[a, b, c]
    T_flat_V = T.transpose(0, 2, 1).reshape(n*m*p*n, m*p)

    start_time = time.time()
    timeout = 110 # Leave 10s margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')

    # Try ranks from 25 to 35
    # Rank 36 is trivial. We want < 36.
    for R in range(25, 36):
        if time.time() - start_time > timeout:
            break
            
        # Multiple random restarts
        for attempt in range(5):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + R * 100 + attempt)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            err = float('inf')
            
            # ALS Iterations
            for it in range(50):
                if time.time() - start_time > timeout:
                    break
                    
                # Update W
                # A_W[(a,b), r] = U[a,r] V[b,r]
                # A_W shape (n*m*m*p, R)
                A_W = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, R)
                # Solve A_W W^T = T_flat_W
                # W^T = A_W^+ T_flat_W
                W_T, residuals, rank_A, s = np.linalg.lstsq(A_W, T_flat_W, rcond=None)
                W = W_T.T
                
                # Update U
                # A_U[(b,c), r] = V[b,r] W[c,r]
                # A_U shape (m*p*p*n, R)
                A_U = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, R)
                # Solve A_U U^T = T_flat_U
                U_T, residuals, rank_A, s = np.linalg.lstsq(A_U, T_flat_U, rcond=None)
                U = U_T.T
                
                # Update V
                # A_V[(a,c), r] = U[a,r] W[c,r]
                # A_V shape (n*m*p*n, R)
                A_V = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, R)
                # Solve A_V V^T = T_flat_V
                V_T, residuals, rank_A, s = np.linalg.lstsq(A_V, T_flat_V, rcond=None)
                V = V_T.T
                
                # Check error
                # T_approx[a,b,c] = sum U[a,r] V[b,r] W[c,r]
                # err = || T - T_approx ||_F^2
                # We can compute this efficiently?
                # T_approx_flat_W = einsum('ar,br,cr->abc', U, V, W).reshape(n*m*m*p, p*n)
                # This is expensive if R is large, but R <= 35, dims are small.
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum((T - T_approx)**2)
                
                if err < 1e-10:
                    break
                
            if err < best_err:
                best_err = err
                if err < 1e-8:
                    best_rank = R
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    # If we found a very low rank, we might stop early if R is small
                    # But we want the absolute minimum, so keep going if time permits
                    # However, if we found rank R, we can break the attempt loop for this R
                    break

    if best_U is not None and best_err < 1e-8:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook rank 36
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

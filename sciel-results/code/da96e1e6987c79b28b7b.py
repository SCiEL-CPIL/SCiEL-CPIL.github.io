import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Target Tensor T flattened
    # T[i, j, k] = 1
    T_target = np.ones((nm, mp, pn)) # Wait, indices are (i,j), (j,k), (k,i)
    # The tensor T is defined on indices (i, j, k) where i in 0..n-1, j in 0..m-1, k in 0..p-1
    # But U is indexed by (i, j) -> flat index i*m + j
    # V is indexed by (j, k) -> flat index j*p + k
    # W is indexed by (k, i) -> flat index k*n + i
    # The contraction sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This produces a tensor of shape (n, m, p) effectively, but flattened indices are mixed.
    # Let's construct the target vector for the flattened tensor of shape (n*m, m*p, p*n) is not quite right.
    # The result of the contraction is a tensor with indices (i, j, k).
    # We want this to be 1 for all valid i, j, k.
    # So we have n*m*p equations.
    
    def residual_func(params, r):
        # params is a flat array of size r * (nm + mp + pn)
        U = params[:r*nm].reshape(r, nm).T # nm x r
        V = params[r*nm:r*nm+r*mp].reshape(r, mp).T # mp x r
        W = params[r*nm+r*mp:r*nm+r*mp+r*pn].reshape(r, pn).T # pn x r
        
        # Compute contraction
        # We need to check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        # Efficient way:
        # For each c, we have a rank-1 tensor U_c (flattened to nm) outer V_c (flattened to mp) outer W_c (flattened to pn)?
        # No, U_c is a vector of size nm. It represents a matrix of size n x m.
        # V_c is a vector of size mp. It represents a matrix of size m x p.
        # W_c is a vector of size pn. It represents a matrix of size p x n.
        # The product U_c V_c W_c is not a simple element-wise product.
        # The term for (i, j, k) is U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i].
        
        # We can vectorize this.
        # Create a tensor of shape (n, m, p, r)
        # U_tensor: n x m x r
        U_t = U.T.reshape(n, m, r)
        V_t = V.T.reshape(m, p, r)
        W_t = W.T.reshape(p, n, r)
        
        # We want sum_r U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
        # This is a tensor contraction.
        # We can do this by iterating over c? Or using einsum.
        # einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        target = np.ones((n, m, p))
        return (result - target).flatten()

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Try to reduce rank
    for r in range(rank_school - 1, 40, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Random init
            params = np.random.randn(r * (nm + mp + pn))
            
            # Scale to reasonable values
            params /= np.sqrt(r)
            
            try:
                res = least_squares(residual_func, params, args=(r,), method='lm', max_nfev=500)
                if res.success or res.cost < 1e-8:
                    # Verify
                    U = res.x[:r*nm].reshape(r, nm).T
                    V = res.x[r*nm:r*nm+r*mp].reshape(r, mp).T
                    W = res.x[r*nm+r*mp:r*nm+r*mp+r*pn].reshape(r, pn).T
                    # Check residual explicitly
                    U_t = U.T.reshape(n, m, r)
                    V_t = V.T.reshape(m, p, r)
                    W_t = W.T.reshape(p, n, r)
                    result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
                    if np.allclose(result, np.ones((n, m, p)), atol=1e-6):
                        best_U, best_V, best_W = U, V, W
                        best_rank = r
                        break # Found better rank, stop trying this r? Or continue to lower?
                        # The task asks for rank-minimal. So we should continue to lower r.
            except Exception:
                pass
        
        # If we found a rank, we continue to try lower ranks.
        # If we didn't find any, we continue.
        
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

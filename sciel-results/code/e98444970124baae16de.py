import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a low-rank decomposition
    for rank in range(14, 25):
        if time.time() > deadline:
            break
        result = _find_decomposition(n, m, p, rank)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _find_decomposition(n, m, p, rank):
    # Shapes
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Number of variables
    n_vars = rank * (size_u + size_v + size_w)
    
    # Define the residual function
    def residuals(x):
        # Reshape x into U, V, W
        u_flat = x[:rank * size_u].reshape(rank, size_u).T
        v_flat = x[rank * size_u : rank * (size_u + size_v)].reshape(rank, size_v).T
        w_flat = x[rank * (size_u + size_v) :].reshape(rank, size_w).T
        
        # Compute the tensor T_reconstructed
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to check if T_reconstructed == 1 for all i,j,k
        
        # Efficient computation:
        # For each r, the term is U[:, r] (size n*m) * V[:, r] (size m*p) * W[:, r] (size p*n)
        # This is a tensor product.
        # We can compute the sum over r.
        
        # Let's compute the error for all (i, j, k)
        # i in 0..n-1, j in 0..m-1, k in 0..p-1
        
        errors = np.zeros((n, m, p))
        
        # Loop over rank terms
        for r in range(rank):
            u_r = u_flat[:, r] # size n*m
            v_r = v_flat[:, r] # size m*p
            w_r = w_flat[:, r] # size p*n
            
            # Reshape to (n, m), (m, p), (p, n)
            u_2d = u_r.reshape(n, m)
            v_2d = v_r.reshape(m, p)
            w_2d = w_r.reshape(p, n)
            
            # The contribution to T[i, j, k] is u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
            # We need to sum this over r.
            
            # To do this efficiently without triple loop in Python:
            # We can use einsum
            # term = np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
            # But doing this in a loop over r is O(rank * n*m*p) which is small.
            
            term = np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
            errors += term
            
        # The target is 1 for all entries
        target = np.ones((n, m, p))
        return (errors - target).flatten()
    
    # Initial guess
    # Random initialization
    np.random.seed(42) # For reproducibility during development, but maybe random is better for search?
    # Let's use a few random restarts
    
    best_result = None
    best_err = float('inf')
    
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random init
        x0 = np.random.randn(n_vars)
        
        # Solve
        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=500)
            if res.cost < best_err:
                best_err = res.cost
                best_x = res.x
        except Exception:
            continue
            
        if best_err < 1e-8:
            # Check if we can round to half-integers
            u_flat = best_x[:rank * size_u].reshape(rank, size_u).T
            v_flat = best_x[rank * size_u : rank * (size_u + size_v)].reshape(rank, size_v).T
            w_flat = best_x[rank * (size_u + size_v) :].reshape(rank, size_w).T
            
            # Round to nearest 0.5
            u_rounded = np.round(u_flat * 2) / 2
            v_rounded = np.round(v_flat * 2) / 2
            w_rounded = np.round(w_flat * 2) / 2
            
            # Verify rounded solution
            errors = np.zeros((n, m, p))
            for r in range(rank):
                u_2d = u_rounded[:, r].reshape(n, m)
                v_2d = v_rounded[:, r].reshape(m, p)
                w_2d = w_rounded[:, r].reshape(p, n)
                term = np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
                errors += term
                
            if np.allclose(errors, np.ones((n, m, p)), atol=1e-9):
                return {
                    "U": u_rounded.tolist(),
                    "V": v_rounded.tolist(),
                    "W": w_rounded.tolist()
                }
                
            # If rounding fails, maybe the real solution is valid but not half-integer?
            # The problem says "integer/half-integer entries".
            # If the solver found a real solution with error 0, but it's not half-integer,
            # it might be a valid solution if the problem allows reals?
            # "integer/half-integer entries" implies the output must be of this type.
            # So we must find a half-integer solution.
            
    return None

def _schoolbook(n, m, p):
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

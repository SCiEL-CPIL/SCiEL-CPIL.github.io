import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 18
    
    # Indices for the multiplication tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # We want sum_c U[u,c] V[v,c] W[w,c] = T[u,v,w]
    
    # Residual function
    # Variables are flattened U, V, W
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    def residuals(x):
        U = x[:n*m*target_rank].reshape(n*m, target_rank)
        V = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        # Compute the tensor contraction
        # Result[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
        # This is expensive to compute naively in a loop if done wrong.
        # We can use einsum: 'uc,vc,wc->uvw'
        # But we only need the 24 non-zero entries of T to be 1, and others 0.
        # Actually, we need the FULL tensor to match.
        # The tensor T is sparse.
        # We can just compute the full tensor and compare to T.
        
        # Efficient computation:
        # Tensor = sum_c outer(U[:,c], V[:,c]) * W[:,c]
        # This is a 3D array.
        # Let's compute it.
        
        # For speed, we can just compute the specific 24 entries that should be 1,
        # and a few others that should be 0?
        # No, we need the whole thing to be zero elsewhere to be a valid decomposition.
        # But if we minimize the norm of the difference, it should work.
        
        # Let's compute the full tensor T_calc
        # T_calc[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
        
        # Using einsum is fast enough for 24x24x24?
        # 24*24*24 = 13824 elements.
        # 18 iterations.
        # 18 * 13824 operations ~ 250k ops. Very fast.
        
        T_calc = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        # Target Tensor T_target
        # We need to construct it or compare on the fly.
        # Constructing it every time is wasteful.
        # But 13824 elements is small.
        
        # Let's pre-compute the indices of non-zero T.
        # Actually, let's just return the difference.
        # We need a global T_target.
        
        return T_calc - T_target

    # Pre-compute T_target
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    # Start time
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial guess
    # Schoolbook is rank 24. We need rank 18.
    # We can initialize with random values, or try to use the first 18 columns of schoolbook?
    # Schoolbook columns are orthogonal. Merging them might not work well.
    # Random initialization with small values is better for optimization.
    
    best_result = None
    best_rank = n * m * p # 24
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback schoolbook
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find rank 18
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale: values around 1.0
        np.random.seed(42 + restart)
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=500)
            # Check if solution is good
            if res.cost < 1e-6:
                # Round to nearest 0.5
                U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                # Verify exactness
                T_calc = np.einsum('uc,vc,wc->uvw', U_round, V_round, W_round)
                if np.allclose(T_calc, T_target, atol=1e-9):
                    best_rank = target_rank
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    break
        except Exception:
            pass

    # Convert to lists for output
    # The prompt asks for integer/half-integer entries.
    # If best_rank is 24, use schoolbook.
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

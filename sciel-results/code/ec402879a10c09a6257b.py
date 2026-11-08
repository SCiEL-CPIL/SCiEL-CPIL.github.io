import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Construct the target tensor T
    # T[I, J, K] where I=i*m+j, J=j*p+k, K=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Pre-compute RHS vectors for ALS
    RHS_U = np.array([T[i, :, :].flatten() for i in range(dim_u)]) # (dim_u, dim_v*dim_w)
    RHS_V = np.array([T[:, j, :].flatten() for j in range(dim_v)]) # (dim_v, dim_u*dim_w)
    RHS_W = np.array([T[:, :, k].flatten() for k in range(dim_w)]) # (dim_w, dim_u*dim_v)

    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T with half-integer entries."""
        # Snap to half-integers
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        
        # Check if snap is close to original (ensures we found a local min near half-integers)
        if not (np.allclose(U, U_h, atol=0.01) and np.allclose(V, V_h, atol=0.01) and np.allclose(W, W_h, atol=0.01)):
            return None
            
        # Verify exactness
        T_approx = np.einsum('ic,jc,kc->ijk', U_h, V_h, W_h)
        if np.allclose(T_approx, T, atol=1e-9):
            return {
                "U": U_h.tolist(),
                "V": V_h.tolist(),
                "W": W_h.tolist()
            }
        return None

    def try_rank(r, max_iters=100, num_restarts=2000, deadline=None):
        best_result = None
        
        for restart in range(num_restarts):
            if deadline and time.time() > deadline:
                break
                
            # Use different random seeds for diversity
            np.random.seed(12345 + restart * 7)
            
            # Initialize with small random values
            # Scale is important. Values are usually small for these decompositions.
            U = np.random.randn(dim_u, r) * 0.5
            V = np.random.randn(dim_v, r) * 0.5
            W = np.random.randn(dim_w, r) * 0.5
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return best_result
                
                # Update U
                M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                try:
                    # Use lstsq for better numerical stability in high condition number cases
                    U_new = np.linalg.lstsq(M_U.T, RHS_U.flatten(), rcond=None)[0].reshape(dim_u, r)
                except:
                    continue
                
                # Update V
                M_V = np.einsum('ic,kc->ikc', U_new, W).reshape(dim_u * dim_w, r)
                try:
                    V_new = np.linalg.lstsq(M_V.T, RHS_V.flatten(), rcond=None)[0].reshape(dim_v, r)
                except:
                    continue
                
                # Update W
                M_W = np.einsum('ic,jc->ijc', U_new, V_new).reshape(dim_u * dim_v, r)
                try:
                    W_new = np.linalg.lstsq(M_W.T, RHS_W.flatten(), rcond=None)[0].reshape(dim_w, r)
                except:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Aggressive snap check every few iterations
                if it % 5 == 0:
                    res = check_decomposition(U, V, W)
                    if res:
                        return res

        return None

    deadline = time.time() + 115 # Leave 5s margin for safety
    
    # Try rank 29 first. This is the target improvement over 30.
    res = try_rank(29, max_iters=50, num_restarts=3000, deadline=deadline)
    if res:
        return res
        
    # Fallback to schoolbook (rank 30) if no better solution is found in time
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

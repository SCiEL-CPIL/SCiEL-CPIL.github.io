import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T as a 3D array or use a function
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # We will try to find a decomposition of rank R
    # Start with likely ranks
    candidate_ranks = [31, 30, 32, 33, 35, 40, 50]
    
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    for R in candidate_ranks:
        if time.time() - start_time > timeout:
            break
            
        # Try multiple random initializations for this rank
        for trial in range(20):
            if time.time() - start_time > timeout:
                break
                
            # Initialize U, V, W with small random values
            # Scaling is important
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1
            
            # ALS iterations
            for it in range(50):
                if time.time() - start_time > timeout:
                    break
                    
                # Update U
                # For each u, U[u, :] = argmin sum_v,w T[u,v,w] - sum_r U[u,r] V[v,r] W[w,r]
                # This is a least squares problem for each u
                # Let M_u be the matrix where M_u[v, r] = sum_w T[u,v,w] W[w,r]
                # Then U[u, :] = M_u @ (V^T V)^{-1} ? No.
                # We want to minimize || T_u - U_u V W^T ||_F^2 ?
                # Actually, the tensor is flattened.
                # T is a 3D array.
                # Let's construct the full tensor T once.
                # T[u, v, w]
                
            # This loop is getting complex to implement ALS efficiently without pre-computing T.
            # Given the small size (10x25x10), we can just build T.
            
    # Let's refactor to build T first, then run ALS.
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    for R in candidate_ranks:
        if time.time() - start_time > timeout:
            break
            
        for trial in range(50):
            if time.time() - start_time > timeout:
                break
                
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # ALS
            for it in range(100):
                if time.time() - start_time > timeout:
                    break
                    
                # Update U
                # For each u, we want to minimize || T[u, :, :] - U[u, :] @ (V * W^T) || ?
                # No, the term is sum_r U[u,r] V[v,r] W[w,r].
                # Let M[u, v, r] = sum_w T[u,v,w] W[w,r].
                # Then we want U[u, r] such that U[u, :] approximates the projection.
                # Specifically, for fixed V, W, the optimal U[u, :] is:
                # U[u, :] = (sum_{v,w} T[u,v,w] W[w,:]^T V[v,:]^T) ... this is getting messy.
                
                # Simpler: 
                # Residual R = T - sum_r U_r V_r W_r
                # Update U: U_new = argmin || T - U V W^T ||
                # This is equivalent to solving for each u:
                # U[u, :] @ (V * W^T) = T[u, :, :]
                # Let A_u = T[u, :, :] reshaped to (dim_v, dim_w)
                # Let B = V * W^T (element-wise product of V and W^T? No)
                # The term is sum_r U[u,r] (V[:,r] * W[:,r]^T)
                # So we are summing outer products of V[:,r] and W[:,r] weighted by U[u,r].
                # Let M_r = V[:,r] * W[:,r]^T (shape dim_v x dim_w)
                # We want to express A_u as sum_r U[u,r] M_r
                # This is a linear system: U[u, :] @ M = A_u
                # Where M is a 3D array (R, dim_v, dim_w) or flattened.
                # Flatten M to (R, dim_v*dim_w).
                # Flatten A_u to (dim_v*dim_w,).
                # Solve U[u, :] @ M_flat = A_u_flat
                # U[u, :] = A_u_flat @ pinv(M_flat)
                
                M_flat = np.zeros((R, dim_v * dim_w))
                for r in range(R):
                    M_flat[r, :] = (V[:, r] * W[:, r]).flatten() # Wait, V[:,r] is dim_v, W[:,r] is dim_w. Outer product?
                    # The term is V[v,r] * W[w,r].
                    # So the matrix is np.outer(V[:,r], W[:,r])
                    M_flat[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                
                # Solve for U
                # U[u, :] is a row vector of length R.
                # A_u_flat is a vector of length dim_v*dim_w.
                # We want U[u, :] such that sum_r U[u,r] M_flat[r, :] = A_u_flat
                # This is M_flat.T @ U[u, :] = A_u_flat ?
                # M_flat is (R, D). A_u_flat is (D,).
                # We want x (length R) such that M_flat.T @ x = A_u_flat.
                # x = pinv(M_flat.T) @ A_u_flat
                
                U_new = np.zeros_like(U)
                for u in range(dim_u):
                    A_u_flat = T[u, :, :].flatten()
                    # Solve M_flat.T @ x = A_u_flat
                    # Use lstsq
                    sol, _, _, _ = np.linalg.lstsq(M_flat.T, A_u_flat, rcond=None)
                    U_new[u, :] = sol
                    
                U = U_new
                
                # Similarly for V and W
                # Update V:
                # For each v, V[v, :] such that sum_r V[v,r] (U[:,r] * W[:,r]^T) = T[:, v, :]
                M_flat_V = np.zeros((R, dim_u * dim_w))
                for r in range(R):
                    M_flat_V[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                    
                V_new = np.zeros_like(V)
                for v in range(dim_v):
                    A_v_flat = T[:, v, :].flatten()
                    sol, _, _, _ = np.linalg.lstsq(M_flat_V.T, A_v_flat, rcond=None)
                    V_new[v, :] = sol
                V = V_new
                
                # Update W:
                # For each w, W[w, :] such that sum_r W[w,r] (U[:,r] * V[:,r]^T) = T[:, :, w]
                M_flat_W = np.zeros((R, dim_u * dim_v))
                for r in range(R):
                    M_flat_W[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                    
                W_new = np.zeros_like(W)
                for w in range(dim_w):
                    A_w_flat = T[:, :, w].flatten()
                    sol, _, _, _ = np.linalg.lstsq(M_flat_W.T, A_w_flat, rcond=None)
                    W_new[w, :] = sol
                W = W_new
                
            # Check error
            # Compute reconstructed tensor
            # Rec = sum_r U[:,r] V[:,r] W[:,r]^T
            # This is expensive to do explicitly for every check.
            # Just check a few random points or use the residual from last step?
            # Let's compute the error on the support of T.
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        val = 0.0
                        for r in range(R):
                            val += U[u, r] * V[v, r] * W[w, r]
                        error += (val - 1.0) ** 2
                        
            # Also check zeros?
            # If the rank is low, it might not satisfy zeros.
            # But if it satisfies ones, it's likely a good candidate.
            
            if error < 1e-6:
                # Round to 0.5
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                exact_error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            val = 0.0
                            for r in range(R):
                                val += U_int[u, r] * V_int[v, r] * W_int[w, r]
                            exact_error += (val - 1.0) ** 2
                            
                # Check a few zero positions
                # u=0, v=1 (i=0,j=0; j=0,k=1 -> w=1*2+0=2)
                # T[0,1,2] should be 0.
                zero_error = 0.0
                # Sample some zeros
                for _ in range(100):
                    u = np.random.randint(0, dim_u)
                    v = np.random.randint(0, dim_v)
                    w = np.random.randint(0, dim_w)
                    # Check if this is a non-zero target
                    # u=(i,j), v=(j',k), w=(k',i')
                    # Non-zero if i=i', j=j', k=k'
                    i1, j1 = divmod(u, m)
                    j2, k1 = divmod(v, p)
                    k2, i2 = divmod(w, n)
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    val = 0.0
                    for r in range(R):
                        val += U_int[u, r] * V_int[v, r] * W_int[w, r]
                    zero_error += val ** 2
                    
                if exact_error < 1e-9 and zero_error < 1e-9:
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    
    # Fallback to schoolbook
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

import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the tensor T
    # T is dim_u x dim_v x dim_w
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Unfold T
    T1 = T.reshape(dim_u, dim_v * dim_w)
    T2 = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
    T3 = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_u, rank_schoolbook))
    V_school = np.zeros((dim_v, rank_schoolbook))
    W_school = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 110 # 110s limit
    
    # Try ranks from 35 down to 23
    for R in range(rank_schoolbook - 1, 22, -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for each rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                # M = W (x) V (Khatri-Rao)
                # M is (dim_v * dim_w) x R
                # M[:, r] = vec(W[:, r] * V[:, r])? No.
                # Khatri-Rao product of W and V (both column vectors for a fixed r)
                # The column r of W x V is the element-wise product of col r of W and col r of V?
                # No, Khatri-Rao product of matrices A (m x n) and B (p x n) is (m*p x n).
                # Column r is A[:, r] * B[:, r] (element-wise).
                # So M[:, r] = (W[:, r] * V[:, r]) flattened?
                # Wait, the indices for T1 are u and (v, w).
                # T1[u, v*w + w] = T[u, v, w].
                # We want T1 = U M^T.
                # M^T is R x (dim_v * dim_w).
                # M^T[r, v*w + w] = W[w, r] * V[v, r].
                # So M[:, r] is a vector of length dim_v * dim_w.
                # M[v*w + w, r] = W[w, r] * V[v, r].
                
                M = np.zeros((dim_v * dim_w, R))
                for r in range(R):
                    # Flatten W[:, r] * V[:, r]?
                    # W is dim_w x R, V is dim_v x R.
                    # We need a vector of length dim_v * dim_w.
                    # Index v*w + w.
                    # For fixed r, M[:, r][v*w + w] = W[w, r] * V[v, r].
                    # This is the Kronecker product V[:, r] (kron) W[:, r]?
                    # Let's check: (V kron W)[v*dim_w + w] = V[v] * W[w].
                    # Yes.
                    M[:, r] = np.kron(V[:, r], W[:, r])
                
                # Solve U = T1 M (M^T M)^-1
                try:
                    U = T1 @ np.linalg.pinv(M)
                except:
                    break
                    
                # Update V
                # T2 = V N^T
                # N[:, r] = np.kron(U[:, r], W[:, r])?
                # T2[v, u*w + w] = T[v, u, w].
                # N^T[r, u*w + w] = U[u, r] * W[w, r].
                # N[:, r][u*w + w] = U[u, r] * W[w, r].
                # This is U[:, r] (kron) W[:, r].
                N = np.zeros((dim_u * dim_w, R))
                for r in range(R):
                    N[:, r] = np.kron(U[:, r], W[:, r])
                try:
                    V = T2 @ np.linalg.pinv(N)
                except:
                    break
                    
                # Update W
                # T3 = W P^T
                # P[:, r] = np.kron(U[:, r], V[:, r])?
                # T3[w, u*v + v] = T[w, u, v].
                # P^T[r, u*v + v] = U[u, r] * V[v, r].
                # P[:, r][u*v + v] = U[u, r] * V[v, r].
                # This is U[:, r] (kron) V[:, r].
                P = np.zeros((dim_u * dim_v, R))
                for r in range(R):
                    P[:, r] = np.kron(U[:, r], V[:, r])
                try:
                    W = T3 @ np.linalg.pinv(P)
                except:
                    break
                    
                # Check convergence
                # Error = || T - sum U V W ||
                # We can check residual on T1
                Res = T1 - U @ M.T
                if np.linalg.norm(Res, 'fro') < 1e-8:
                    converged = True
                    break
                    
            if converged:
                # Verify full tensor
                T_recon = np.zeros((dim_u, dim_v, dim_w))
                for r in range(R):
                    T_recon += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
                if np.allclose(T_recon, T, atol=1e-8):
                    best_rank = R
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break # Found a valid rank R, stop searching for lower R?
                    # Actually, we want the MINIMAL rank. So we should continue to lower R.
                    # But if we found R, we can try R-1.
                    # The loop goes downwards, so we will try lower ranks next.
                    # We should NOT break the rank loop, only the restart loop.
                    # But we found a valid one. Let's update best and continue.
                    # Actually, if we found R, we can break the restart loop and continue to R-1.
                    pass 
                
    return best_result

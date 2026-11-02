import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T of size (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1? 
    # Wait, the indices in the task description:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Output C = sum_r U[:,r] outer V[:,r] ... no.
    # The decomposition is: T = sum_r U_r \otimes V_r \otimes W_r
    # Where T is the matrix multiplication tensor.
    # T_{a, b, c} = 1 if the term contributes.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # For a given i,j,k, T[i*m+j, j*p+k, k*n+i] = 1.
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Function to verify a decomposition
    def verify(U, V, W, tol=1e-9):
        # U: (size_u, R), V: (size_v, R), W: (size_w, R)
        R = U.shape[1]
        # Compute sum_r U[:,r] outer V[:,r] outer W[:,r]
        # This is a tensor of size size_u x size_v x size_w
        # We can check specific entries or the whole thing.
        # Since T is sparse, check only non-zeros?
        # Or compute the full tensor and compare.
        # size_u=9, size_v=9, size_w=9. Total 729 entries.
        # R is small.
        # Tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
        # This might be slow if R is large, but R<27, so it's fine.
        
        # Actually, we can just check the 27 non-zero entries of T.
        # But we need to ensure the other 702 entries are 0.
        # So we must check the whole tensor or use a norm.
        
        # Efficient check:
        # C = U @ V.T? No.
        # The decomposition implies:
        # For each r, U_r, V_r, W_r are vectors.
        # The tensor is sum_r np.outer(np.outer(U_r, V_r), W_r)
        
        # Let's use einsum for simplicity and speed (9x9x9 is tiny).
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    # ALS Search
    def search_rank(R, max_iters=100, tol=1e-10):
        # Initialize U, V, W randomly
        # Use a deterministic seed for reproducibility? 
        # Or random.
        np.random.seed(42)
        U = np.random.randn(size_u, R)
        V = np.random.randn(size_v, R)
        W = np.random.randn(size_w, R)
        
        for it in range(max_iters):
            # Update U
            # T_{a, b, c} = sum_r U_{a,r} V_{b,r} W_{c,r}
            # Fix V, W.
            # Z_{b, c, r} = V_{b,r} W_{c,r}
            # T_{a, b, c} = sum_r U_{a,r} Z_{b, c, r}
            # Reshape T to (size_u, size_v * size_w)
            # Reshape Z to (size_v * size_w, R)
            T_mat = T.reshape(size_u, size_v * size_w)
            Z_mat = (V * W[None, :, :]).reshape(size_v * size_w, R) # V is (size_v, R), W is (size_w, R)
            # V[:, None, :] * W[None, :, :] -> (size_v, size_w, R)
            # Reshape to (size_v * size_w, R)
            Z_mat = (V[:, None, :] * W[None, :, :]).reshape(size_v * size_w, R)
            
            # Solve T_mat = U_new @ Z_mat.T ?
            # T_mat[a, bc] = sum_r U_new[a, r] Z_mat[bc, r]
            # So T_mat = U_new @ Z_mat.T
            # U_new = T_mat @ pinv(Z_mat.T)
            # Or U_new = pinv(Z_mat) @ T_mat.T .T ?
            # Let's use lstsq.
            # U_new is (size_u, R).
            # We want to minimize || T_mat - U_new @ Z_mat.T ||_F
            # This is equivalent to minimizing || Z_mat @ U_new.T - T_mat.T ||_F
            # So we solve Z_mat @ X = T_mat.T for X (R x size_u)
            # X = lstsq(Z_mat, T_mat.T)
            U_new, _, _, _ = np.linalg.lstsq(Z_mat, T_mat.T, rcond=None)
            U = U_new.T
            
            # Update V
            # T_{a, b, c} = sum_r U_{a,r} V_{b,r} W_{c,r}
            # Fix U, W.
            # Y_{a, c, r} = U_{a,r} W_{c,r}
            # T_{a, b, c} = sum_r V_{b,r} Y_{a, c, r}
            # Reshape T to (size_v, size_u * size_w)
            # Reshape Y to (size_u * size_w, R)
            T_mat_v = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
            Y_mat = (U[:, None, :] * W[None, :, :]).reshape(size_u * size_w, R)
            
            # T_mat_v = V_new @ Y_mat.T
            # V_new = lstsq(Y_mat, T_mat_v.T).T
            V_new, _, _, _ = np.linalg.lstsq(Y_mat, T_mat_v.T, rcond=None)
            V = V_new.T
            
            # Update W
            # T_{a, b, c} = sum_r U_{a,r} V_{b,r} W_{c,r}
            # Fix U, V.
            # X_{a, b, r} = U_{a,r} V_{b,r}
            # T_{a, b, c} = sum_r W_{c,r} X_{a, b, r}
            # Reshape T to (size_w, size_u * size_v)
            # Reshape X to (size_u * size_v, R)
            T_mat_w = T.transpose(2, 0, 1).reshape(size_w, size_u * size_v)
            X_mat = (U[:, None, :] * V[None, :, :]).reshape(size_u * size_v, R)
            
            # T_mat_w = W_new @ X_mat.T
            # W_new = lstsq(X_mat, T_mat_w.T).T
            W_new, _, _, _ = np.linalg.lstsq(X_mat, T_mat_w.T, rcond=None)
            W = W_new.T
            
            # Check convergence
            if it % 10 == 0 or it == max_iters - 1:
                if verify(U, V, W, tol):
                    return U, V, W
        
        return None

    # Try ranks from 26 down to 19
    best_rank = 27
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    for R in range(26, 18, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            res = search_rank(R, max_iters=50, tol=1e-9)
            if res is not None:
                U, V, W = res
                # Final verification with tighter tolerance
                if verify(U, V, W, tol=1e-8):
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            # If we found a lower rank, we can try to go lower
            # But the loop continues downwards automatically
            pass

    if best_rank < 27 and best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook rank 27
        rank = 27
        U = [[0] * rank for _ in range(size_u)]
        V = [[0] * rank for _ in range(size_v)]
        W = [[0] * rank for _ in range(size_w)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

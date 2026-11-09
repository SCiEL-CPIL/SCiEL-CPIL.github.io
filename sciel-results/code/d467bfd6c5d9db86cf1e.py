import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T[I, J, K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i for valid (i,j,k)
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Pre-compute RHS vectors for ALS updates
    RHS_U = T.reshape(dim_u, dim_v * dim_w)  # (dim_u, dim_v*dim_w)
    RHS_V = np.zeros((dim_v, dim_u * dim_w))
    for j in range(dim_v):
        RHS_V[j] = T[:, j, :].flatten()
    RHS_W = np.zeros((dim_w, dim_u * dim_v))
    for k in range(dim_w):
        RHS_W[k] = T[:, :, k].flatten()

    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety

    def compute_error(U, V, W, r):
        """Compute the error of the decomposition."""
        T_approx = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_approx += U[:, c][:, np.newaxis, np.newaxis] * V[:, c][np.newaxis, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
        return np.linalg.norm(T_approx - T)

    def verify_decomposition(U, V, W, r):
        """Verify exact decomposition and check half-integer constraint."""
        T_approx = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_approx += U[:, c][:, np.newaxis, np.newaxis] * V[:, c][np.newaxis, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
        
        if not np.allclose(T_approx, T, atol=1e-10):
            return False
        
        # Check if entries are half-integers (within tolerance)
        U_check = np.round(U * 2) / 2
        V_check = np.round(V * 2) / 2
        W_check = np.round(W * 2) / 2
        
        if not np.allclose(U, U_check, atol=1e-8):
            return False
        if not np.allclose(V, V_check, atol=1e-8):
            return False
        if not np.allclose(W, W_check, atol=1e-8):
            return False
        
        # Final exact verification with rounded values
        T_approx2 = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_approx2 += U_check[:, c][:, np.newaxis, np.newaxis] * V_check[:, c][np.newaxis, :, np.newaxis] * W_check[:, c][np.newaxis, np.newaxis, :]
        
        if not np.allclose(T_approx2, T, atol=1e-10):
            return False
        
        return True

    def try_round_and_verify(U, V, W, r):
        """Try rounding to half-integers and verify."""
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        if verify_decomposition(U_half, V_half, W_half, r):
            return U_half, V_half, W_half
        return None

    def local_search_half_integer(U, V, W, r, max_steps=100):
        """Local search over half-integer neighbors."""
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        current_error = compute_error(U_half, V_half, W_half, r)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if current_error < 1e-10:
                if verify_decomposition(U_half, V_half, W_half, r):
                    return U_half, V_half, W_half
                break
            
            improved = False
            # Try perturbing one entry at a time
            for mat_name, mat in [('U', U_half), ('V', V_half), ('W', W_half)]:
                if time.time() > deadline:
                    break
                for i in range(mat.shape[0]):
                    if time.time() > deadline:
                        break
                    for j in range(mat.shape[1]):
                        old_val = mat[i, j]
                        # Try +/- 0.5
                        for delta in [0.5, -0.5]:
                            new_val = old_val + delta
                            if mat_name == 'U':
                                U_half[i, j] = new_val
                                err = compute_error(U_half, V_half, W_half, r)
                                U_half[i, j] = old_val
                            elif mat_name == 'V':
                                V_half[i, j] = new_val
                                err = compute_error(U_half, V_half, W_half, r)
                                V_half[i, j] = old_val
                            else:
                                W_half[i, j] = new_val
                                err = compute_error(U_half, V_half, W_half, r)
                                W_half[i, j] = old_val
                            
                            if err < current_error - 1e-12:
                                if mat_name == 'U':
                                    U_half[i, j] = new_val
                                elif mat_name == 'V':
                                    V_half[i, j] = new_val
                                else:
                                    W_half[i, j] = new_val
                                current_error = err
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if current_error < 1e-10:
            if verify_decomposition(U_half, V_half, W_half, r):
                return U_half, V_half, W_half
        return None

    def try_rank(r, max_iters=200, tol=1e-9, num_restarts=50):
        """
        Attempts to find a decomposition of rank r using vectorized ALS.
        Returns (U, V, W) if successful, else None.
        """
        best_error = np.inf
        best_result = None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Use different initialization strategies
            seed = 777 + restart
            np.random.seed(seed)
            
            # Mix of random and structured initializations
            if restart % 3 == 0:
                U = np.random.randn(dim_u, r) * 0.5
                V = np.random.randn(dim_v, r) * 0.5
                W = np.random.randn(dim_w, r) * 0.5
            elif restart % 3 == 1:
                # Initialize with small integers/half-integers
                U = (np.random.randint(0, 4, (dim_u, r)) - 1.5) * 0.5
                V = (np.random.randint(0, 4, (dim_v, r)) - 1.5) * 0.5
                W = (np.random.randint(0, 4, (dim_w, r)) - 1.5) * 0.5
            else:
                # Sparse initialization
                U = np.random.randn(dim_u, r) * 0.5
                V = np.random.randn(dim_v, r) * 0.5
                W = np.random.randn(dim_w, r) * 0.5
                # Zero out some entries
                mask_u = np.random.rand(dim_u, r) < 0.5
                mask_v = np.random.rand(dim_v, r) < 0.5
                mask_w = np.random.rand(dim_w, r) < 0.5
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0
            
            prev_error = np.inf
            stagnant_count = 0
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                try:
                    M_U_pinv = np.linalg.pinv(M_U)
                    U_new = (M_U_pinv @ RHS_U.T).T
                except np.linalg.LinAlgError:
                    continue
                
                # Update V
                M_V = np.einsum('ic,kc->ikc', U_new, W).reshape(dim_u * dim_w, r)
                try:
                    M_V_pinv = np.linalg.pinv(M_V)
                    V_new = (M_V_pinv @ RHS_V.T).T
                except np.linalg.LinAlgError:
                    continue
                
                # Update W
                M_W = np.einsum('ic,jc->ijc', U_new, V_new).reshape(dim_u * dim_v, r)
                try:
                    M_W_pinv = np.linalg.pinv(M_W)
                    W_new = (M_W_pinv @ RHS_W.T).T
                except np.linalg.LinAlgError:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                error = compute_error(U, V, W, r)
                
                if error < best_error:
                    best_error = error
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if error < tol:
                    # Polish
                    for polish_it in range(30):
                        if time.time() > deadline:
                            break
                        M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                        try:
                            M_U_pinv = np.linalg.pinv(M_U)
                            U = (M_U_pinv @ RHS_U.T).T
                        except:
                            break
                        
                        M_V = np.einsum('ic,kc->ikc', U, W).reshape(dim_u * dim_w, r)
                        try:
                            M_V_pinv = np.linalg.pinv(M_V)
                            V = (M_V_pinv @ RHS_V.T).T
                        except:
                            break
                        
                        M_W = np.einsum('ic,jc->ijc', U, V).reshape(dim_u * dim_v, r)
                        try:
                            M_W_pinv = np.linalg.pinv(M_W)
                            W = (M_W_pinv @ RHS_W.T).T
                        except:
                            break
                        
                        error = compute_error(U, V, W, r)
                        if error < 1e-12:
                            break
                    
                    # Try direct verification
                    if verify_decomposition(U, V, W, r):
                        return U, V, W
                    
                    # Try rounding
                    result = try_round_and_verify(U, V, W, r)
                    if result is not None:
                        return result
                    
                    # Try local search
                    result = local_search_half_integer(U, V, W, r, max_steps=50)
                    if result is not None:
                        return result
                    
                    if best_result is None:
                        best_result = (U.copy(), V.copy(), W.copy())
                    break
                
                # Check for stagnation
                if error > prev_error * 0.999:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                prev_error = error
                
                if stagnant_count > 15:
                    break
        
        # If we have a best result, try to verify it
        if best_result is not None and best_error < 1e-4:
            U, V, W = best_result
            result = try_round_and_verify(U, V, W, r)
            if result is not None:
                return result
            
            result = local_search_half_integer(U, V, W, r, max_steps=100)
            if result is not None:
                return result
        
        return None

    # Try ranks from 28 down to 20 (focus on 28 first since inspirations found it)
    for r in range(28, 19, -1):
        if time.time() > deadline:
            break
        
        # Use more restarts for lower ranks
        if r <= 26:
            num_restarts = 80
        else:
            num_restarts = 50
        
        result = try_rank(r, num_restarts=num_restarts)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook (rank 30)
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

import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = 4, 5, 6
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    # Build target tensor
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(result, atol=1e-8):
        U = np.array(result["U"], dtype=np.float64)
        V = np.array(result["V"], dtype=np.float64)
        W = np.array(result["W"], dtype=np.float64)
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def als_decompose(target, rank, max_iter=100, tol=1e-12, seed=0):
        """
        ALS decomposition for the matrix multiplication tensor.
        """
        np.random.seed(seed)
        
        # Initialization
        # Random initialization often works better than truncated schoolbook for finding non-trivial structures
        U = np.random.randn(N, rank)
        V = np.random.randn(M, rank)
        W = np.random.randn(P, rank)
        
        # Scale to avoid large initial errors
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = np.inf
        for iteration in range(max_iter):
            if time_remaining() < 10:
                break
            
            # Update U: fix V, W
            # Target U[a, :] such that sum_r U[a, r] (V^T W^T)_{r, b, c} = T[a, b, c]
            # Let H_U[b, c, r] = sum_{a'} ... no, H_U is the "core" tensor contracted with V and W
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed V, W, this is a linear system in U.
            # Reshape T to (N, M*P)
            # Reshape (V^T W^T) to (M*P, rank) ?
            # V is (M, rank), W is (P, rank).
            # V^T is (rank, M). W^T is (rank, P).
            # The term sum_r V[b, r] W[c, r] is the element (b, c) of the matrix V @ W.T
            # Let K = V @ W.T  (shape M x P)
            # Then T[a, b, c] = sum_r U[a, r] K[b, c] is NOT correct.
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Let's group indices b, c.
            # For each r, the contribution to T[:, :, :] is U[:, r] outer V[:, r] outer W[:, r].
            # This is not a simple matrix multiplication.
            
            # Correct ALS step:
            # T_{a b c} = \sum_r U_{a r} V_{b r} W_{c r}
            # Fix V, W. We want to minimize || T - \sum_r U_{a r} V_{b r} W_{c r} ||_F^2 w.r.t U.
            # This is equivalent to:
            # For each a, the vector U[a, :] is the least squares solution to:
            # \sum_r U_{a r} (V_{b r} W_{c r}) \approx T_{a b c}
            # Let H_U[b, c, r] = V[b, r] W[c, r]
            # Then we solve \sum_r H_U[b, c, r] U_{a r} = T_{a b c}
            # This is a linear system: A_U U_a^T = T_a^T
            # Where A_U is a matrix of size (M*P) x rank.
            # A_U[(b, c), r] = V[b, r] W[c, r]
            
            # Construct A_U
            # V is (M, rank), W is (P, rank)
            # We want A_U such that A_U[(b,c), r] = V[b, r] W[c, r]
            # This is the Khatri-Rao product or similar.
            # Let's use einsum to construct the matrix A_U of shape (M*P, rank)
            # Actually, we can solve it more efficiently.
            # T is (N, M, P). Reshape to (N, M*P).
            # A_U is (M*P, rank).
            # U is (N, rank).
            # We want U @ A_U.T to approximate T_reshaped.
            # So U = T_reshaped @ pinv(A_U.T)
            
            # Construct A_U
            # V[:, r] is a vector of length M. W[:, r] is a vector of length P.
            # The r-th column of A_U is the outer product V[:, r] outer W[:, r] flattened.
            # A_U = np.zeros((M*P, rank))
            # for r in range(rank):
            #     A_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Vectorized construction of A_U
            # V is (M, rank), W is (P, rank)
            # We want to compute V[b, r] * W[c, r] for all b, c, r.
            # This is a tensor of shape (M, P, rank).
            # A_U_tensor = np.einsum('br,cr->bcr', V, W)
            # A_U = A_U_tensor.reshape(M*P, rank)
            
            A_U_tensor = np.einsum('br,cr->bcr', V, W)
            A_U = A_U_tensor.reshape(M * P, rank)
            T_flat = target.reshape(N, M * P)
            
            try:
                # Solve U @ A_U.T = T_flat
                # U = T_flat @ pinv(A_U.T)
                # Using lstsq for stability
                U_new, _, _, _ = np.linalg.lstsq(A_U, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V
            A_V_tensor = np.einsum('ar,cr->acr', U, W)
            A_V = A_V_tensor.reshape(N * P, rank)
            T_flat_v = target.transpose(1, 0, 2).reshape(M, N * P)
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(A_V, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Update W
            A_W_tensor = np.einsum('ar,br->abr', U, V)
            A_W = A_W_tensor.reshape(N * M, rank)
            T_flat_w = target.transpose(2, 0, 1).reshape(P, N * M)
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(A_W, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check convergence
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_approx - target)
            
            if abs(prev_err - err) < tol and err < 1e-9:
                break
            prev_err = err
        
        # Final check
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_approx, target, atol=1e-8):
            return U, V, W
        return None

    def try_rank(r, num_trials=10):
        if time_remaining() < 15:
            return None
        for trial in range(num_trials):
            if time_remaining() < 10:
                break
            U, V, W = als_decompose(target, r, max_iter=50, seed=12345 + trial * 7)
            if U is not None:
                return {"U": U, "V": V, "W": W}
        return None

    # Search for lower ranks
    # Start from 119 down to 100
    for r in range(best_rank - 1, max(best_rank - 30, 95), -1):
        if time_remaining() < 20:
            break
        result = try_rank(r, num_trials=5)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            # If we find a low rank, we might want to try to go even lower
            # But for now, accept the first valid one found in descending order
            # Actually, let's keep searching a bit more if we have time
            # But usually, the first found is close to the optimum for this size
            break 

    # If no lower rank found, return schoolbook
    if best_rank == n * m * p:
        # Try a few more random trials for rank 119 just in case
        if time_remaining() > 30:
            result = try_rank(119, num_trials=20)
            if result is not None:
                best = result
                best_rank = 119

    # Final verification
    if not verify(best, atol=1e-7):
        best = schoolbook()
        best_rank = n * m * p

    # Round to nearest half-integer if close
    U_out = np.array(best["U"])
    V_out = np.array(best["V"])
    W_out = np.array(best["W"])
    
    U_rounded = np.round(U_out * 2) / 2
    V_rounded = np.round(V_out * 2) / 2
    W_rounded = np.round(W_out * 2) / 2
    
    result_rounded = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    if verify(result_rounded, atol=1e-6):
        best = result_rounded

    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}

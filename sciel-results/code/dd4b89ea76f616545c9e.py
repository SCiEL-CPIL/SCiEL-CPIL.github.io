# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Build target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-8, rtol=1e-8)

    def snap_to_half_integer(U, V, W):
        """Snap entries to nearest half-integer if close."""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        if verify(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return U, V, W

    def try_rank(R, rng):
        """Attempt to find a decomposition of rank R."""
        # Random initialization
        U = rng.normal(size=(n * m, R))
        V = rng.normal(size=(m * p, R))
        W = rng.normal(size=(p * n, R))
        
        # Scale to match magnitude
        U /= np.sqrt(R)
        V /= np.sqrt(R)
        W /= np.sqrt(R)
        
        lr = 0.01
        max_iters = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Compute current tensor
            T_hat = np.einsum('ic,jc,kc->ijk', U, V, W)
            residual = T - T_hat
            
            # Check convergence
            if np.linalg.norm(residual) < 1e-8:
                U_s, V_s, W_s = snap_to_half_integer(U, V, W)
                if verify(U_s, V_s, W_s):
                    return U_s, V_s, W_s
                return U, V, W

            # Compute gradients
            grad_U = 2 * np.einsum('ijk,jc,kc->ic', residual, V, W)
            grad_V = 2 * np.einsum('ijk,ic,kc->jc', residual, U, W)
            grad_W = 2 * np.einsum('ijk,ic,jc->kc', residual, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            if it % 50 == 0:
                lr *= 0.9

        return None

    # Initialize with schoolbook as fallback
    U0 = [[0] * base_rank for _ in range(n * m)]
    V0 = [[0] * base_rank for _ in range(m * p)]
    W0 = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
                
    best_U, best_V, best_W = None, None, None
    best_rank = base_rank
    rng = np.random.default_rng(42)
    
    # Try to find lower rank
    for R in range(base_rank - 1, base_rank - 5, -1):
        if time.time() > deadline:
            break
        for attempt in range(3):
            if time.time() > deadline:
                break
            res = try_rank(R, rng)
            if res is not None:
                U_found, V_found, W_found = res
                if verify(U_found, V_found, W_found):
                    best_U, best_V, best_W = U_found, V_found, W_found
                    best_rank = R
                    break
        if best_rank < base_rank:
            break

    if best_U is not None:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a greedy search to reduce the rank from 26.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    def build_rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        terms = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for r, (u_terms, v_terms, w_terms) in enumerate(terms):
            for (i, j, val) in u_terms:
                U[idx_u(i, j), r] = val
            for (j, k, val) in v_terms:
                V[idx_v(j, k), r] = val
            for (k, i, val) in w_terms:
                W[idx_w(k, i), r] = val
        
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W, R
    
    U, V, W, R = build_rank26()
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
    
    if not verify(U, V, W, R):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Greedy search to reduce rank
    # Try to remove terms and solve for the rest
    # This is complex, so we'll try a simple random search for rank 25
    if time.time() < deadline:
        np.random.seed(42)
        for target_R in [25, 24]:
            if time.time() > deadline - 15:
                break
            for attempt in range(100):
                if time.time() > deadline - 15:
                    break
                U_try = np.zeros((n * m, target_R))
                V_try = np.zeros((m * p, target_R))
                W_try = np.zeros((p * n, target_R))
                
                # Initialize with Strassen
                terms = [
                    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                    ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
                    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
                    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
                    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
                    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
                ]
                
                for r, (u_terms, v_terms, w_terms) in enumerate(terms):
                    if r >= target_R:
                        break
                    for (i, j, val) in u_terms:
                        U_try[idx_u(i, j), r] = val
                    for (j, k, val) in v_terms:
                        V_try[idx_v(j, k), r] = val
                    for (k, i, val) in w_terms:
                        W_try[idx_w(k, i), r] = val
                
                # Fill remaining with random
                for r in range(7, target_R):
                    n_u = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    n_v = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    n_w = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    
                    u_positions = np.random.choice(n * m, size=n_u, replace=False)
                    v_positions = np.random.choice(m * p, size=n_v, replace=False)
                    w_positions = np.random.choice(p * n, size=n_w, replace=False)
                    
                    for pos in u_positions:
                        U_try[pos, r] = np.random.choice([-1, 1])
                    for pos in v_positions:
                        V_try[pos, r] = np.random.choice([-1, 1])
                    for pos in w_positions:
                        W_try[pos, r] = np.random.choice([-1, 1])
                
                if verify(U_try, V_try, W_try, target_R):
                    best_U, best_V, best_W, best_R = U_try.copy(), V_try.copy(), W_try.copy(), target_R
                    break
    
    if not verify(best_U, best_V, best_W, best_R):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

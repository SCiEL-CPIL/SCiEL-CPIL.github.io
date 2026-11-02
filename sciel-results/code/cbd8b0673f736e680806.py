def _try_fast_search(n, m, p, start_time, deadline=DEADLINE_SECONDS):
    best = _build_schoolbook(n, m, p)
    schoolbook_rank = n * m * p
    np.random.seed(42)
    
    for target_rank in range(schoolbook_rank - 1, 40, -1):
        if time.time() - start_time > deadline - 5:
            break
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 5:
                break
            
            # Generate random U, V with small integer entries
            # Strategy: sparse with values in {-1, 0, 1}
            density_u = np.random.choice([0.1, 0.2, 0.3, 0.5])
            density_v = np.random.choice([0.1, 0.2, 0.3, 0.5])
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            U[np.random.random((n*m, target_rank)) < density_u] = np.random.choice([-1, 1], size=(n*m, target_rank))
            V[np.random.random((m*p, target_rank)) < density_v] = np.random.choice([-1, 1], size=(m*p, target_rank))
            
            # Zero out entries where random was >= density
            U[U == 0] = 0  # Already zero
            V[V == 0] = 0
            
            # Actually, let me do this more carefully
            U_mask = np.random.random((n*m, target_rank)) < density_u
            V_mask = np.random.random((m*p, target_rank)) < density_v
            U = np.where(U_mask, np.random.choice([-1, 1], size=(n*m, target_rank)), 0.0)
            V = np.where(V_mask, np.random.choice([-1, 1], size=(m*p, target_rank)), 0.0)
            
            W = _fast_w_compute(U, V, n, m, p)
            if W is None:
                continue
            
            if _fast_verify(U, V, W, n, m, p):
                # Convert to lists
                U_list = [[float(U[i, c]) for c in range(target_rank)] for i in range(n * m)]
                V_list = [[float(V[i, c]) for c in range(target_rank)] for i in range(m * p)]
                W_list = [[float(W[i, c]) for c in range(target_rank)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best

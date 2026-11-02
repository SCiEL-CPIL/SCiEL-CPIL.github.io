def _search_rank(n, m, p, r, field_p, num_u, num_v, num_w, num_eq):
    random.seed(42)
    np.random.seed(42)
    
    max_trials = 500
    
    # Structured Initialization
    # Fix first min(r, num_u) columns of U to Identity
    # Fix first min(r, num_v) columns of V to Identity
    
    for trial in range(max_trials):
        U = np.zeros((num_u, r), dtype=int)
        V = np.zeros((num_v, r), dtype=int)
        
        # Set Identity parts
        if r >= num_u:
            U[:num_u, :num_u] = np.eye(num_u, dtype=int)
        else:
            U[:, :r] = np.eye(num_u, r, dtype=int) # Not possible if r < num_u for full rank, but let's just fill random
            
        if r >= num_v:
            V[:num_v, :num_v] = np.eye(num_v, dtype=int)
        else:
            V[:, :r] = np.eye(num_v, r, dtype=int)
            
        # Fill remaining with random sparse values
        prob = 0.5 if field_p == 2 else 0.3
        
        for i in range(num_u):
            for c in range(min(r, num_u), r):
                if random.random() < prob:
                    U[i, c] = random.randint(1, field_p - 1)
                    
        for j in range(num_v):
            for c in range(min(r, num_v), r):
                if random.random() < prob:
                    V[j, c] = random.randint(1, field_p - 1)
                    
        # Build A and b
        A = np.zeros((num_eq, num_w * r), dtype=int)
        b = np.zeros(num_eq, dtype=int)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m * p):
                    j_prime = jp // p
                    k = jp % p
                    for kp in range(p * n):
                        k_prime = kp // n
                        i_prime = kp % n
                        
                        if i == i_prime and j == j_prime and k == k_prime:
                            b[eq_idx] = 1
                            
                        for c in range(r):
                            coeff = (U[i * m + j, c] * V[j_prime * p + k, c]) % field_p
                            if coeff != 0:
                                w_idx = (k_prime * n + i_prime) * r + c
                                A[eq_idx, w_idx] = (A[eq_idx, w_idx] + coeff) % field_p
                                
                        eq_idx += 1
                        
        W_flat = _solve_over_gf(A, b, field_p)
        
        if W_flat is None:
            continue
            
        W = W_flat.reshape((num_w, r))
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                lhs = 0
                                for c in range(r):
                                    lhs = (lhs + U[i * m + j, c] * V[j_prime * p + k, c] * W[k_prime * n + i_prime, c]) % field_p
                                rhs = 1 if (j == j_prime and i == i_prime and k == k_prime) else 0
                                if lhs != rhs:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
            
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

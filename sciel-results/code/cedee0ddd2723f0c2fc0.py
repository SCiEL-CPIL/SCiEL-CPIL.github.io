"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    import random
    import time
    
    # Dimensions
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Target tensor indices
    # We need to identify which (u, v, w) triples have target value 1, and which have 0.
    # Target T[u, v, w] = 1 if there exist i,j,k such that:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # This implies:
    # j = u % m, i = u // m
    # k = v % p, j' = v // p -> must match j
    # i' = w % n, k' = w // n -> must match k and i
    # So a triple (u,v,w) is "active" (target 1) if:
    # (u % m) == (v // p) and (v % p) == (w // n) and (w % n) == (u // m)
    
    active_triples = []
    inactive_triples = []
    for u in range(num_u):
        for v in range(num_v):
            for w in range(num_w):
                i = u // m
                j = u % m
                j_check = v // p
                k = v % p
                k_check = w // n
                i_check = w % n
                if j == j_check and k == k_check and i == i_check:
                    active_triples.append((u, v, w))
                else:
                    inactive_triples.append((u, v, w))
    
    # Precompute contributions for fast error update
    # We will search for rank R.
    # We use a local search on half-integers.
    # Variables are stored in flat lists or 2D lists.
    
    def find_decomposition(R, max_iters=50000, seed=42):
        random.seed(seed)
        
        # Initialize with random half-integers in [-2, 2]
        # Values are integers representing * 0.5
        U = [[random.randint(-4, 4) for _ in range(R)] for _ in range(num_u)]
        V = [[random.randint(-4, 4) for _ in range(R)] for _ in range(num_v)]
        W = [[random.randint(-4, 4) for _ in range(R)] for _ in range(num_w)]
        
        # Compute initial errors
        # error[u, v, w] = sum_r U[u,r] V[v,r] W[w,r] - Target
        # We store errors in a flat array or dict? 
        # Given small size (324 triples), a list of errors is fine.
        # But we need to map (u,v,w) to index.
        
        # Let's just compute total error and update incrementally.
        # To update efficiently, we need the current sum for each triple.
        # S[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        
        S = {}
        for u in range(num_u):
            for v in range(num_v):
                for w in range(num_w):
                    s = 0
                    for r in range(R):
                        s += U[u][r] * V[v][r] * W[w][r]
                    S[(u, v, w)] = s
        
        def get_error():
            err = 0
            for t in active_triples:
                u, v, w = t
                err += (S[t] - 2) ** 2 # Target is 1.0, values are half-ints (int/2). 
                # Wait, if U,V,W are integers representing half-ints, then product is (int/2)^3 = int/8.
                # Target is 1 = 8/8.
                # So we should compare S[t] (which is sum of ints) to 8?
                # Let's rescale.
                # If U,V,W entries are integers $u, v, w$, the actual value is $u/2 \cdot v/2 \cdot w/2 = uvw/8$.
                # Sum is $\sum uvw / 8$.
                # Target is 1 or 0.
                # So $\sum uvw$ should be 8 for active, 0 for inactive.
                # Error = (S[t] - 8)^2 for active, (S[t])^2 for inactive.
                pass
            for t in inactive_triples:
                u, v, w = t
                err += (S[t]) ** 2
            return err

        # Re-implement get_error with the scaling logic
        def compute_error():
            err = 0
            for t in active_triples:
                err += (S[t] - 8) ** 2
            for t in inactive_triples:
                err += (S[t]) ** 2
            return err

        current_error = compute_error()
        
        if current_error == 0:
            # Convert back to floats (half-integers)
            U_res = [[x * 0.5 for x in row] for row in U]
            V_res = [[x * 0.5 for x in row] for row in V]
            W_res = [[x * 0.5 for x in row] for row in W]
            return U_res, V_res, W_res

        best_error = current_error
        best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        
        # Local Search
        for _ in range(max_iters):
            # Pick a random variable
            # Matrix choice
            mat_choice = random.choice([0, 1, 2])
            if mat_choice == 0:
                row = random.randint(0, num_u - 1)
                col = random.randint(0, R - 1)
                old_val = U[row][col]
                new_val = old_val + random.choice([-1, 1])
                # Update S
                # S[u, v, w] += delta * V[v, r] * W[w, r]
                delta = new_val - old_val
                r = col
                for v in range(num_v):
                    for w in range(num_w):
                        t = (row, v, w)
                        S[t] += delta * V[v][r] * W[w][r]
                
                new_error = compute_error()
                
                if new_error < current_error:
                    U[row][col] = new_val
                    current_error = new_error
                    if current_error == 0:
                        U_res = [[x * 0.5 for x in row] for row in U]
                        V_res = [[x * 0.5 for x in row] for row in V]
                        W_res = [[x * 0.5 for x in row] for row in W]
                        return U_res, V_res, W_res
                    if current_error < best_error:
                        best_error = current_error
                        best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                else:
                    # Revert
                    for v in range(num_v):
                        for w in range(num_w):
                            t = (row, v, w)
                            S[t] -= delta * V[v][r] * W[w][r]
                            
            elif mat_choice == 1:
                row = random.randint(0, num_v - 1)
                col = random.randint(0, R - 1)
                old_val = V[row][col]
                new_val = old_val + random.choice([-1, 1])
                delta = new_val - old_val
                r = col
                for u in range(num_u):
                    for w in range(num_w):
                        t = (u, row, w)
                        S[t] += delta * U[u][r] * W[w][r]
                
                new_error = compute_error()
                
                if new_error < current_error:
                    V[row][col] = new_val
                    current_error = new_error
                    if current_error == 0:
                        U_res = [[x * 0.5 for x in row] for row in U]
                        V_res = [[x * 0.5 for x in row] for row in V]
                        W_res = [[x * 0.5 for x in row] for row in W]
                        return U_res, V_res, W_res
                    if current_error < best_error:
                        best_error = current_error
                        best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                else:
                    for u in range(num_u):
                        for w in range(num_w):
                            t = (u, row, w)
                            S[t] -= delta * U[u][r] * W[w][r]
                            
            else:
                row = random.randint(0, num_w - 1)
                col = random.randint(0, R - 1)
                old_val = W[row][col]
                new_val = old_val + random.choice([-1, 1])
                delta = new_val - old_val
                r = col
                for u in range(num_u):
                    for v in range(num_v):
                        t = (u, v, row)
                        S[t] += delta * U[u][r] * V[v][r]
                
                new_error = compute_error()
                
                if new_error < current_error:
                    W[row][col] = new_val
                    current_error = new_error
                    if current_error == 0:
                        U_res = [[x * 0.5 for x in row] for row in U]
                        V_res = [[x * 0.5 for x in row] for row in V]
                        W_res = [[x * 0.5 for x in row] for row in W]
                        return U_res, V_res, W_res
                    if current_error < best_error:
                        best_error = current_error
                        best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                else:
                    for u in range(num_u):
                        for v in range(num_v):
                            t = (u, v, row)
                            S[t] -= delta * U[u][r] * V[v][r]

        # If no exact solution found, check if best_state is close? 
        # The problem requires EXACT decomposition.
        # So we must return None if not exact.
        return None

    # Try ranks from lower bound
    # Lower bound for M_{2,3,3} is 9.
    # Known rank is 12? Let's try 12, 13, 14, 15, 16, 17.
    # Start with 12.
    
    for R in range(12, 19):
        res = find_decomposition(R, max_iters=20000, seed=R)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook if search fails
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(num_u)]
    V = [[0] * rank_school for _ in range(num_v)]
    W = [[0] * rank_school for _ in range(num_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def construct():
    deadline = _deadline_guard(140.0)
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find a lower rank solution using local search
    # We try ranks 17, 16, 15
    for target_rank in [17, 16, 15]:
        if time.time() > deadline:
            break
        sol = _local_search(n, m, p, target_rank, deadline)
        if sol is not None:
            best = sol
            best_rank = target_rank
            break

    return best

def _local_search(n, m, p, rank, deadline):
    """
    Attempts to find a rank 'rank' decomposition using stochastic local search.
    Returns the solution dict if found, None otherwise.
    """
    rng = random.Random(int(time.time() * 1000) % 10000) # Pseudo-random seed for variety
    
    # Initialize with random sparse matrices
    def init_random():
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        vals = [0, 1, -1, 2, -2] # Doubled values: 0, 0.5, -0.5, 1, -1
        for i in range(n*m):
            for c in range(rank):
                if rng.random() < 0.3:
                    U[i][c] = rng.choice(vals)
        for j in range(m*p):
            for c in range(rank):
                if rng.random() < 0.3:
                    V[j][c] = rng.choice(vals)
        for k in range(p*n):
            for c in range(rank):
                if rng.random() < 0.3:
                    W[k][c] = rng.choice(vals)
        return U, V, W

    U, V, W = init_random()
    
    def compute_residuals(U, V, W):
        res = [[[0]*p for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i*m + j
                    vj = j*p + k
                    wk = k*n + i
                    for c in range(rank):
                        s += U[ui][c] * V[vj][c] * W[wk][c]
                    res[i][j][k] = 8 - s
        return res

    def error_metric(res):
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = res[i][j][k]
                    err += r*r
        return err

    current_res = compute_residuals(U, V, W)
    current_err = error_metric(current_res)
    
    steps = 0
    max_steps = 50000
    while time.time() < deadline and steps < max_steps:
        steps += 1
        unsat = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if current_res[i][j][k] != 0:
                        unsat.append((i,j,k))
        
        if not unsat:
            return {"U": U, "V": V, "W": W}
            
        i, j, k = rng.choice(unsat)
        c = rng.randint(0, rank-1)
        mat_idx = rng.randint(0, 2)
        
        if mat_idx == 0:
            idx = i*m + j
            old_val = U[idx][c]
            new_val = old_val + rng.choice([-2, -1, 1, 2])
            
            delta_err = 0
            for kp in range(p):
                delta_sum = (new_val - old_val) * V[j*p+kp][c] * W[kp*n+i][c]
                old_r = current_res[i][j][kp]
                new_r = old_r - delta_sum
                delta_err += new_r*new_r - old_r*old_r
                
            if delta_err < 0:
                U[idx][c] = new_val
                for kp in range(p):
                    delta_sum = (new_val - old_val) * V[j*p+kp][c] * W[kp*n+i][c]
                    current_res[i][j][kp] -= delta_sum
                current_err += delta_err
                continue
        
        elif mat_idx == 1:
            idx = j*p + k
            old_val = V[idx][c]
            new_val = old_val + rng.choice([-2, -1, 1, 2])
            
            delta_err = 0
            for ip in range(n):
                delta_sum = (new_val - old_val) * U[ip*m+j][c] * W[k*n+ip][c]
                old_r = current_res[ip][j][k]
                new_r = old_r - delta_sum
                delta_err += new_r*new_r - old_r*old_r
                
            if delta_err < 0:
                V[idx][c] = new_val
                for ip in range(n):
                    delta_sum = (new_val - old_val) * U[ip*m+j][c] * W[k*n+ip][c]
                    current_res[ip][j][k] -= delta_sum
                current_err += delta_err
                continue

        else:
            idx = k*n + i
            old_val = W[idx][c]
            new_val = old_val + rng.choice([-2, -1, 1, 2])
            
            delta_err = 0
            for jp in range(m):
                delta_sum = (new_val - old_val) * U[i*m+jp][c] * V[jp*p+k][c]
                old_r = current_res[i][jp][k]
                new_r = old_r - delta_sum
                delta_err += new_r*new_r - old_r*old_r
                
            if delta_err < 0:
                W[idx][c] = new_val
                for jp in range(m):
                    delta_sum = (new_val - old_val) * U[i*m+jp][c] * V[jp*p+k][c]
                    current_res[i][jp][k] -= delta_sum
                current_err += delta_err
                continue

    return None
# EVOLVE-BLOCK-END

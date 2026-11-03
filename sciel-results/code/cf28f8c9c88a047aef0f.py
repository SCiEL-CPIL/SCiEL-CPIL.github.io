<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower ranks
    # Try a set of target ranks in descending order
    target_ranks = [120, 110, 100, 90, 80, 70, 60, 50, 48, 45, 40]
    
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try both random and structured initializations
        for init_type in ['random', 'structured']:
            if time.time() > deadline - 10:
                break
                
            for seed in range(5):
                if time.time() > deadline - 5:
                    break
                    
                if init_type == 'random':
                    U_opt, V_opt, W_opt, err = _try_random_init(
                        n, m, p, R, max_iters=150, lr=0.01, seed=seed, deadline=deadline
                    )
                else:
                    U_opt, V_opt, W_opt, err = _try_structured_init(
                        n, m, p, R, max_iters=150, lr=0.01, seed=seed, deadline=deadline
                    )
                    
                if U_opt is not None:
                    best_rank = R
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    found = True
                    print(f"Found valid decomposition with rank {R} using {init_type} init")
                    break # Move to next lower rank
            
            if found:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower ranks - use a more aggressive search
    target_ranks = [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32]
    
    for R in target_ranks:
        if time.time() > deadline - 25:
            break
            
        found = False
        # Try multiple random initializations
        for seed in range(8):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(seed)
            init_U = np.random.randn(n*m, R) * 0.1
            init_V = np.random.randn(m*p, R) * 0.1
            init_W = np.random.randn(p*n, R) * 0.1
            
            U_opt, V_opt, W_opt, err = _try_optimize(
                n, m, p, R, init_U, init_V, init_W,
                max_iters=200, lr=0.01, deadline=deadline
            )
            
            if U_opt is not None:
                best_rank = R
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                found = True
                print(f"Found valid decomposition with rank {R}")
                break
        
        # Also try structured initialization if not found
        if not found and time.time() < deadline - 15:
            np.random.seed(123)
            U_struct = U_school[:, :R].copy()
            V_struct = V_school[:, :R].copy()
            W_struct = W_school[:, :R].copy()
            U_struct += np.random.randn(*U_struct.shape) * 0.05
            V_struct += np.random.randn(*V_struct.shape) * 0.05
            W_struct += np.random.randn(*W_struct.shape) * 0.05
            
            U_opt, V_opt, W_opt, err = _try_optimize(
                n, m, p, R, U_struct, V_struct, W_struct,
                max_iters=200, lr=0.01, deadline=deadline
            )
            
            if U_opt is not None:
                best_rank = R
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                found = True
                print(f"Found valid decomposition with rank {R} (structured)")
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE

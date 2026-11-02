def _try_find_2x2x3_rank10(deadline):
    """
    Attempt to find a rank 10 decomposition for 2x2x3.
    Uses random initialization and local search.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # We will optimize U, V, W
    # Initialize with random small integers
    np.random.seed(42)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Simple local search
    # We try to minimize sum (val - target)^2
    
    max_iters = 10000 # Adjust based on time, roughly 10ms per iter? 100s? 
    # Let's be conservative: 2000 iterations to be safe, or check time.
    
    current_U = np.random.randint(-2, 3, (n*m, rank))
    current_V = np.random.randint(-2, 3, (m*p, rank))
    current_W = np.random.randint(-2, 3, (p*n, rank))
    
    # Calculate initial error
    def calc_err(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    target = 1.0 if i == k else 0.0
                    err += (val - target) ** 2
        return err

    curr_err = calc_err(current_U, current_V, current_W)
    if curr_err < best_err:
        best_err = curr_err
        best_U = current_U.copy()
        best_V = current_V.copy()
        best_W = current_W.copy()
        if best_err < 1e-6 and _verify_decomposition(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W

    for it in range(5000):
        if time.time() > deadline - 5:
            break
        
        # Random restart every 1000 iterations
        if it % 1000 == 0:
            current_U = np.random.randint(-2, 3, (n*m, rank))
            current_V = np.random.randint(-2, 3, (m*p, rank))
            current_W = np.random.randint(-2, 3, (p*n, rank))
            curr_err = calc_err(current_U, current_V, current_W)
            if curr_err < best_err:
                best_err = curr_err
                best_U = current_U.copy()
                best_V = current_V.copy()
                best_W = current_W.copy()
                if best_err < 1e-6 and _verify_decomposition(best_U, best_V, best_W, n, m, p):
                    return best_U, best_V, best_W
            continue
            
        # Local search step
        mat_choice = np.random.randint(3)
        if mat_choice == 0:
            r, c = np.random.randint(n*m), np.random.randint(rank)
            old_val = current_U[r, c]
            current_U[r, c] += np.random.choice([-1, 1])
            new_err = calc_err(current_U, current_V, current_W)
            if new_err < curr_err:
                curr_err = new_err
                if curr_err < best_err:
                    best_err = curr_err
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                    if best_err < 1e-6 and _verify_decomposition(best_U, best_V, best_W, n, m, p):
                        return best_U, best_V, best_W
            else:
                current_U[r, c] = old_val
        elif mat_choice == 1:
            r, c = np.random.randint(m*p), np.random.randint(rank)
            old_val = current_V[r, c]
            current_V[r, c] += np.random.choice([-1, 1])
            new_err = calc_err(current_U, current_V, current_W)
            if new_err < curr_err:
                curr_err = new_err
                if curr_err < best_err:
                    best_err = curr_err
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                    if best_err < 1e-6 and _verify_decomposition(best_U, best_V, best_W, n, m, p):
                        return best_U, best_V, best_W
            else:
                current_V[r, c] = old_val
        else:
            r, c = np.random.randint(p*n), np.random.randint(rank)
            old_val = current_W[r, c]
            current_W[r, c] += np.random.choice([-1, 1])
            new_err = calc_err(current_U, current_V, current_W)
            if new_err < curr_err:
                curr_err = new_err
                if curr_err < best_err:
                    best_err = curr_err
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                    if best_err < 1e-6 and _verify_decomposition(best_U, best_V, best_W, n, m, p):
                        return best_U, best_V, best_W
            else:
                current_W[r, c] = old_val

    return None

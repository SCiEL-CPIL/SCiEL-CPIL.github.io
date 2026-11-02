def _als_search(rank, time_limit):
    """Search for a rank-r decomposition using ALS."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Build the target tensor (sparse)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # For each a = i*m+j, the non-zero entries are:
    # T[a, j*p+k, k*n+i] = 1 for k=0..p-1
    tensor_entries = []  # List of (a, b, c) tuples
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # For each a, store the list of (b, c) pairs
    a_to_bc = {}
    for a, b, c in tensor_entries:
        if a not in a_to_bc:
            a_to_bc[a] = []
        a_to_bc[a].append((b, c))
    
    # Initialize U, V, W randomly
    np.random.seed(42)
    U = np.random.randn(num_a, rank)
    V = np.random.randn(num_b, rank)
    W = np.random.randn(num_c, rank)
    
    # ALS iterations
    for iteration in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        for a in range(num_a):
            if a not in a_to_bc:
                continue
            bc_list = a_to_bc[a]
            # Build the 8×rank matrix K where K[k, r] = V[b_k, r] * W[c_k, r]
            K = np.array([V[b, :] * W[c, :] for b, c in bc_list])
            # Solve K @ U[a,:]^T = 1 (minimum norm)
            # U[a,:]^T = K^T @ (K @ K^T)^{-1} @ 1
            try:
                KKt = K @ K.T
                rhs = np.ones(len(bc_list))
                U[a, :] = np.linalg.solve(KKt, K.T @ rhs)
            except np.linalg.LinAlgError:
                # Use pseudoinverse
                U[a, :] = np.linalg.lstsq(K, rhs, rcond=None)[0]
        
        # Update V
        for b in range(num_b):
            # Find all (a, c) such that T[a, b, c] = 1
            # b = j*p+k, so j = b // p, k = b % p
            j = b // p
            k = b % p
            # For each i, a = i*m+j, c = k*n+i
            ac_list = []
            for i in range(n):
                a = i * m + j
                c = k * n + i
                ac_list.append((a, c))
            # Build the n×rank matrix K where K[i, r] = U[a_i, r] * W[c_i, r]
            K = np.array([U[a, :] * W[c, :] for a, c in ac_list])
            rhs = np.ones(len(ac_list))
            try:
                KKt = K @ K.T
                V[b, :] = np.linalg.solve(KKt, K.T @ rhs)
            except np.linalg.LinAlgError:
                V[b, :] = np.linalg.lstsq(K, rhs, rcond=None)[0]
        
        # Update W
        for c in range(num_c):
            # Find all (a, b) such that T[a, b, c] = 1
            # c = k*n+i, so k = c // n, i = c % n
            k = c // n
            i = c % n
            # For each j, a = i*m+j, b = j*p+k
            ab_list = []
            for j in range(m):
                a = i * m + j
                b = j * p + k
                ab_list.append((a, b))
            # Build the m×rank matrix K where K[j, r] = U[a_j, r] * V[b_j, r]
            K = np.array([U[a, :] * V[b, :] for a, b in ab_list])
            rhs = np.ones(len(ab_list))
            try:
                KKt = K @ K.T
                W[c, :] = np.linalg.solve(KKt, K.T @ rhs)
            except np.linalg.LinAlgError:
                W[c, :] = np.linalg.lstsq(K, rhs, rcond=None)[0]
        
        # Check convergence
        if time.time() - start_time > time_limit - 5:
            break
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

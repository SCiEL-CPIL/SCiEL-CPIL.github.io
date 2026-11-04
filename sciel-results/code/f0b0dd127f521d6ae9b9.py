def _try_rank_reduction(U, V, W, n=4, m=4, p=8, max_removals=20, time_limit=100):
    """Try to reduce rank by removing columns that don't affect the decomposition."""
    import time
    start = time.time()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Test matrices
    np.random.seed(42)
    A = np.random.randint(-3, 4, size=(n, m)).astype(np.float64)
    B = np.random.randint(-3, 4, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    
    def check_validity(U, V, W):
        rank = U.shape[1]
        C_computed = np.zeros((n, p), dtype=np.float64)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] = s
        return np.max(np.abs(C_computed - C_expected)) < 1e-10
    
    removed = []
    for attempt in range(max_removals):
        if time.time() - start > time_limit:
            break
        # Try removing each remaining column
        current_rank = U.shape[1]
        found = False
        for col in range(current_rank):
            if col in removed:
                continue
            # Try removing column col
            mask = [i for i in range(current_rank) if i != col]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if check_validity(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                removed.append(col)
                found = True
                break
        if not found:
            break
    
    return U, V, W, len(removed)

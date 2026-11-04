def _als_optimize(U, V, W, n, m, p, max_iters=200, tol=1e-10, time_limit=10.0):
    """
    Optimize a rank-r decomposition using alternating least squares.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Returns optimized (U, V, W) as lists of lists.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    start_time = time.time()
    prev_loss = np.inf
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
                
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        for a in range(n * m):
            rows = []
            for (ai, bi, ci) in indices:
                if ai == a:
                    rows.append(V_arr[bi] * W_arr[ci])
            if rows:
                M = np.array(rows)
                try:
                    U_arr[a] = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for b in range(m * p):
            rows = []
            for (ai, bi, ci) in indices:
                if bi == b:
                    rows.append(U_arr[ai] * W_arr[ci])
            if rows:
                M = np.array(rows)
                try:
                    V_arr[b] = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        for c in range(p * n):
            rows = []
            for (ai, bi, ci) in indices:
                if ci == c:
                    rows.append(U_arr[ai] * V_arr[bi])
            if rows:
                M = np.array(rows)
                try:
                    W_arr[c] = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        
        # Compute loss
        loss = 0.0
        for (a, b, c) in indices:
            val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
            loss += (val - 1.0) ** 2
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

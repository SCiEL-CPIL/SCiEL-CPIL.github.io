<<<<<<< SEARCH
    terms = build_strassen()
    rank = len(terms)
    
    # Check if we have time to try something better
    elapsed = time.time() - start_time
    if elapsed < 60:
        # Try to find a better decomposition using linear algebra
        # This is a placeholder for more sophisticated search
        pass
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
=======
    terms = build_strassen()
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    # Verify correctness with a quick test
    def verify_decomp(U, V, W, rank):
        np.random.seed(123)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_actual = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    u_prod = sum(U[i * m + j][c] * A_test[i, j] for j in range(m))
                    v_prod = sum(V[j * p + k][c] * B_test[j, k] for j in range(m))
                    total += u_prod * v_prod * W[k * n + i][c]
                C_actual[i, k] = total
        return np.allclose(C_actual, C_expected, atol=1e-10)
    
    if not verify_decomp(U, V, W, rank):
        # Fallback to original partitioning if verification fails
        terms = []
        
        def add_strassen_orig(i_off, j_off, k_off):
            terms.append((
                {(i_off, j_off): 1, (i_off+1, j_off+1): 1},
                {(j_off, k_off): 1, (j_off+1, k_off+1): 1},
                {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
            ))
            terms.append((
                {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
                {(j_off, k_off): 1},
                {(k_off, i_off+1): 1, (k_off+1, i_off+1): -1}
            ))
            terms.append((
                {(i_off, j_off): 1},
                {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1},
                {(k_off+1, i_off): 1, (k_off+1, i_off+1): 1}
            ))
            terms.append((
                {(i_off+1, j_off+1): 1},
                {(j_off+1, k_off): 1, (j_off, k_off): -1},
                {(k_off, i_off): 1, (k_off, i_off+1): 1}
            ))
            terms.append((
                {(i_off, j_off): 1, (i_off, j_off+1): 1},
                {(j_off+1, k_off+1): 1},
                {(k_off, i_off): -1, (k_off+1, i_off): 1}
            ))
            terms.append((
                {(i_off+1, j_off): 1, (i_off, j_off): -1},
                {(j_off, k_off): 1, (j_off, k_off+1): 1},
                {(k_off+1, i_off+1): 1}
            ))
            terms.append((
                {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
                {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
                {(k_off, i_off): 1}
            ))
        
        def add_schoolbook_orig(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        terms.append((
                            {(i, j): 1},
                            {(j, k): 1},
                            {(k, i): 1}
                        ))
        
        add_strassen_orig(0, 0, 0)
        add_strassen_orig(0, 0, 2)
        add_schoolbook_orig(0, 2, 0, 2, 4, 5)
        add_strassen_orig(0, 2, 0)
        add_strassen_orig(0, 2, 2)
        add_schoolbook_orig(0, 2, 2, 4, 4, 5)
        add_schoolbook_orig(0, 2, 4, 5, 0, 5)
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE


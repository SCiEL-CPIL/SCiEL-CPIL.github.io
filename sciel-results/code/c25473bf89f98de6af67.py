def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
    nonlocal col
    i_count = i_end - i_start
    j_count = j_end - j_start
    k_count = k_end - k_start
    
    if i_count == 1 or j_count == 1 or k_count == 1:
        # Rank 1: use a single term
        i0 = i_start
        j0 = j_start
        k0 = k_start
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col] = 1
                    V[idx_v][col] = 1
                    W[idx_w][col] = 1
        col += 1
    else:
        # Full schoolbook: one term per (i,j,k)
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col] = 1
                    V[idx_v][col] = 1
                    W[idx_w][col] = 1
                    col += 1

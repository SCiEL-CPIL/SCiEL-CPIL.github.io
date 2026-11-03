"""Tensor decomposition for 2x5x6 matrix multiplication - schoolbook baseline."""
S = [2, 5, 6]

def construct():
    rows_a, mid, cols_b = S
    total_rank = rows_a * mid * cols_b
    mat_u = [[0] * total_rank for _ in range(rows_a * mid)]
    mat_v = [[0] * total_rank for _ in range(mid * cols_b)]
    mat_w = [[0] * total_rank for _ in range(cols_b * rows_a)]
    
    idx = 0
    for row_i in range(rows_a):
        for mid_j in range(mid):
            for col_k in range(cols_b):
                mat_u[row_i * mid + mid_j][idx] = 1
                mat_v[mid_j * cols_b + col_k][idx] = 1
                mat_w[col_k * rows_a + row_i][idx] = 1
                idx += 1
    
    return {"U": mat_u, "V": mat_v, "W": mat_w}

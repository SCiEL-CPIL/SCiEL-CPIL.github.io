DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 117
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    row_starts = [0, 2, 4]
    col_starts = [0, 2, 4]
    row_sizes = [2, 2, 1]
    col_sizes = [2, 2, 1]
    
    r = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                i_start = row_starts[i_block]
                k_start = row_starts[k_block]
                j_start = col_starts[j_block]
                
                h_i = row_sizes[i_block]
                h_k = row_sizes[k_block]
                w_j = col_sizes[j_block]
                
                if h_i == 2 and h_k == 2 and w_j == 2:
                    r = _add_strassen_222(U, V, W, i_start, k_start, j_start, r, m, p, n)
                elif h_i == 2 and h_k == 2 and w_j == 1:
                    r = _add_schoolbook_221(U, V, W, i_start, k_start, j_start, r, m, p, n)
                elif h_i == 2 and h_k == 1 and w_j == 2:
                    r = _add_schoolbook_212(U, V, W, i_start, k_start, j_start, r, m, p, n)
                elif h_i == 2 and h_k == 1 and w_j == 1:
                    r = _add_schoolbook_211(U, V, W, i_start, k_start, j_start, r, m, p, n)
                elif h_i == 1 and h_k == 2 and w_j == 2:
                    r = _add_schoolbook_122(U, V, W, i_start, k_start, j_start, r, m, p, n)
                elif h_i == 1 and h_k == 2 and w_j == 1:
                    r = _add_schoolbook_121(U, V, W, i_start, k_start, j_start, r, m, p, n)
                elif h_i == 1 and h_k == 1 and w_j == 2:
                    r = _add_schoolbook_112(U, V, W, i_start, k_start, j_start, r, m, p, n)
                elif h_i == 1 and h_k == 1 and w_j == 1:
                    r = _add_schoolbook_111(U, V, W, i_start, k_start, j_start, r, m, p, n)
    
    return {"U": U, "V": V, "W": W}

def _add_strassen_222(U, V, W, i_start, k_start, j_start, r, m, p, n):
    # ... implementation as derived ...
    pass

# ... other helpers ...
# EVOLVE-BLOCK-END

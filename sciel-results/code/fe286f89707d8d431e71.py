DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook_222(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i_off + i][col] = 1
                    V[(j_off + j) * p + (k_off + k)][col] = 1
                    W[(k_off + k) * n + (i_off + i)][col] = 1
                    col += 1
    
    def add_schoolbook_221(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for j in range(2):
                U[i_off + i][col] = 1
                V[(j_off + j) * p + k_off][col] = 1
                W[k_off * n + (i_off + i)][col] = 1
                col += 1
    
    add_schoolbook_222(0, 0, 0)
    add_schoolbook_222(0, 0, 2)
    add_schoolbook_222(0, 2, 0)
    add_schoolbook_222(0, 2, 2)
    add_schoolbook_221(0, 0, 4)
    add_schoolbook_221(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}

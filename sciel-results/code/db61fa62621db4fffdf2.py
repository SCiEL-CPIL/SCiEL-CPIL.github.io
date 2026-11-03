"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Partition:
    # A rows (n=3): [0,1], [2]
    # B cols (p=7): [0,1], [2,3], [4,5], [6]
    # Middle (m=5): [0,1], [2,3], [4]
    
    row_blocks = [[0, 1], [2]]
    col_blocks = [[0, 1], [2, 3], [4, 5], [6]]
    mid_blocks = [[0, 1], [2, 3], [4]]
    
    U = {}  # (i, j) -> list of (col_idx, val)
    V = {}  # (j, k) -> list of (col_idx, val)
    W = {}  # (k, i) -> list of (col_idx, val)
    
    col = 0
    
    def add_U(i, j, val):
        key = (i, j)
        if key not in U:
            U[key] = []
        U[key].append((col, val))
    
    def add_V(j, k, val):
        key = (j, k)
        if key not in V:
            V[key] = []
        V[key].append((col, val))
    
    def add_W(k, i, val):
        key = (k, i)
        if key not in W:
            W[key] = []
        W[key].append((col, val))
    
    for rb in row_blocks:
        for cb in col_blocks:
            for mb in mid_blocks:
                r_size = len(rb)
                c_size = len(cb)
                m_size = len(mb)
                
                if r_size == 2 and c_size == 2 and m_size == 2:
                    # Strassen decomposition
                    r0, r1 = rb
                    c0, c1 = cb
                    m0, m1 = mb
                    
                    # Term 1: M1
                    add_U(r0, m0, 1); add_U(r1, m1, 1)
                    add_V(m0, c0, 1); add_V(m1, c1, 1)
                    add_W(c0, r0, 1); add_W(c1, r1, 1)
                    col += 1
                    
                    # Term 2: M2
                    add_U(r1, m0, 1); add_U(r1, m1, 1)
                    add_V(m0, c0, 1)
                    add_W(c0, r1, 1)
                    col += 1
                    
                    # Term 3: M3
                    add_U(r0, m0, 1)
                    add_V(m0, c1, 1); add_V(m1, c1, -1)
                    add_W(c1, r0, 1)
                    col += 1
                    
                    # Term 4: M4
                    add_U(r1, m1, 1)
                    add_V(m1, c0, 1); add_V(m0, c0, -1)
                    add_W(c0, r1, 1)
                    col += 1
                    
                    # Term 5: M5
                    add_U(r0, m0, 1); add_U(r0, m1, 1)
                    add_V(m1, c1, 1)
                    add_W(c1, r0, 1)
                    col += 1
                    
                    # Term 6: M6
                    add_U(r1, m0, 1); add_U(r0, m0, -1)
                    add_V(m0, c0, 1); add_V(m0, c1, 1)
                    add_W(c0, r1, 1); add_W(c1, r1, 1)
                    col += 1
                    
                    # Term 7: M7
                    add_U(r0, m1, 1); add_U(r1, m1, -1)
                    add_V(m1, c0, 1); add_V(m1, c1, 1)
                    add_W(c0, r0, 1); add_W(c1, r0, 1)
                    col += 1
                else:
                    # Schoolbook
                    for i in rb:
                        for j in mb:
                            for k in cb:
                                add_U(i, j, 1)
                                add_V(j, k, 1)
                                add_W(k, i, 1)
                                col += 1
    
    rank = col
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for c_idx, val in entries:
            U_mat[i * m + j][c_idx] = val
    
    for (j, k), entries in V.items():
        for c_idx, val in entries:
            V_mat[j * p + k][c_idx] = val
    
    for (k, i), entries in W.items():
        for c_idx, val in entries:
            W_mat[k * n + i][c_idx] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END

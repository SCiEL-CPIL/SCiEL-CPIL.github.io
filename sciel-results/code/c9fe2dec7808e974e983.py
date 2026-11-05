import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition 5 into 3, 2
    # Block 0: indices 0, 1, 2 (size 3)
    # Block 1: indices 3, 4 (size 2)
    
    parts = [3, 2]
    offsets = [0, 3]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to create zero vectors
    def zeros():
        return [0]*25, [0]*25, [0]*25

    # Iterate over blocks
    for a in range(2):
        for b in range(2):
            for c in range(2):
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if a == 0 and b == 0 and c == 0:
                    # 3x3x3 BCDFR Decomposition
                    # Indices are local 0,1,2
                    # Global i = off_a + u, j = off_b + v, k = off_c + w
                    
                    # Factor 1
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+i)] = 1
                        v[(off_b+i)*5 + (off_c+i)] = 1
                        w[(off_c+i)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 2
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+i)] = 1
                        v[(off_b+i)*5 + (off_c+(i+1)%3)] = 1
                        w[(off_c+(i+1)%3)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 3
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+i)] = 1
                        v[(off_b+i)*5 + (off_c+(i+2)%3)] = 1
                        w[(off_c+(i+2)%3)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 4
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+(i+1)%3)] = 1
                        v[(off_b+(i+1)%3)*5 + (off_c+i)] = 1
                        w[(off_c+i)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 5
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+(i+1)%3)] = 1
                        v[(off_b+(i+1)%3)*5 + (off_c+(i+1)%3)] = 1
                        w[(off_c+(i+1)%3)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 6
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+(i+1)%3)] = 1
                        v[(off_b+(i+1)%3)*5 + (off_c+(i+2)%3)] = 1
                        w[(off_c+(i+2)%3)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 7
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+(i+2)%3)] = 1
                        v[(off_b+(i+2)%3)*5 + (off_c+i)] = 1
                        w[(off_c+i)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 8
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+(i+2)%3)] = 1
                        v[(off_b+(i+2)%3)*5 + (off_c+(i+1)%3)] = 1
                        w[(off_c+(i+1)%3)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 9
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+(i+2)%3)] = 1
                        v[(off_b+(i+2)%3)*5 + (off_c+(i+2)%3)] = 1
                        w[(off_c+(i+2)%3)*5 + (off_a+i)] = 1
                    add_col(u, v, w)
                    
                    # Factor 10
                    u, v, w = zeros()
                    u[(off_a+0)*5 + (off_b+0)] = 1
                    v[(off_b+0)*5 + (off_c+0)] = 1
                    w[(off_c+0)*5 + (off_a+0)] = 1
                    add_col(u, v, w)
                    
                    # Factor 11
                    u, v, w = zeros()
                    u[(off_a+1)*5 + (off_b+1)] = 1
                    v[(off_b+1)*5 + (off_c+1)] = 1
                    w[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # Factor 12
                    u, v, w = zeros()
                    u[(off_a+2)*5 + (off_b+2)] = 1
                    v[(off_b+2)*5 + (off_c+2)] = 1
                    w[(off_c+2)*5 + (off_a+2)] = 1
                    add_col(u, v, w)
                    
                    # Factor 13
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+0)*5 + (off_b+i)] = 1
                        v[(off_b+2)*5 + (off_c+i)] = 1
                        w[(off_c+i)*5 + (off_a+0)] = 1
                    add_col(u, v, w)
                    
                    # Factor 14
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+1)*5 + (off_b+i)] = 1
                        v[(off_b+i)*5 + (off_c+2)] = 1
                        w[(off_c+2)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # Factor 15
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+2)*5 + (off_b+i)] = 1
                        v[(off_b+0)*5 + (off_c+i)] = 1
                        w[(off_c+i)*5 + (off_a+2)] = 1
                    add_col(u, v, w)
                    
                    # Factor 16
                    u, v, w = zeros()
                    for i in range(3):
                        u[(off_a+i)*5 + (off_b+0)] = 1
                        v[(off_b+0)*5 + (off_c+i)] = 1
                        w[(off_c+i)*5 + (off_a+i)] = 1
                    add_col(u, v, w)

                elif a == 1 and b == 1 and c == 1:
                    # 2x2x2 Strassen Decomposition
                    # Local indices 0, 1
                    # Global i = off_a + u, etc.
                    
                    # Factor 0: M1
                    u, v, w = zeros()
                    u[(off_a+0)*5 + (off_b+0)] = 1
                    u[(off_a+1)*5 + (off_b+1)] = 1
                    v[(off_b+0)*5 + (off_c+0)] = 1
                    v[(off_b+1)*5 + (off_c+1)] = 1
                    w[(off_c+0)*5 + (off_a+0)] = 1
                    w[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # Factor 1: M2
                    u, v, w = zeros()
                    u[(off_a+1)*5 + (off_b+0)] = 1
                    u[(off_a+1)*5 + (off_b+1)] = 1
                    v[(off_b+0)*5 + (off_c+0)] = 1
                    w[(off_c+0)*5 + (off_a+1)] = 1
                    w[(off_c+1)*5 + (off_a+1)] = -1
                    add_col(u, v, w)
                    
                    # Factor 2: M3
                    u, v, w = zeros()
                    u[(off_a+0)*5 + (off_b+0)] = 1
                    v[(off_b+0)*5 + (off_c+1)] = 1
                    v[(off_b+1)*5 + (off_c+1)] = -1
                    w[(off_c+1)*5 + (off_a+0)] = 1
                    w[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # Factor 3: M4
                    u, v, w = zeros()
                    u[(off_a+1)*5 + (off_b+1)] = 1
                    v[(off_b+1)*5 + (off_c+0)] = 1
                    v[(off_b+0)*5 + (off_c+0)] = -1
                    w[(off_c+0)*5 + (off_a+0)] = 1
                    w[(off_c+0)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # Factor 4: M5
                    u, v, w = zeros()
                    u[(off_a+0)*5 + (off_b+0)] = 1
                    u[(off_a+0)*5 + (off_b+1)] = 1
                    v[(off_b+1)*5 + (off_c+1)] = 1
                    w[(off_c+0)*5 + (off_a+0)] = -1
                    w[(off_c+1)*5 + (off_a+0)] = 1
                    add_col(u, v, w)
                    
                    # Factor 5: M6
                    u, v, w = zeros()
                    u[(off_a+1)*5 + (off_b+0)] = 1
                    u[(off_a+0)*5 + (off_b+0)] = -1
                    v[(off_b+0)*5 + (off_c+0)] = 1
                    v[(off_b+0)*5 + (off_c+1)] = 1
                    w[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # Factor 6: M7
                    u, v, w = zeros()
                    u[(off_a+0)*5 + (off_b+1)] = 1
                    u[(off_a+1)*5 + (off_b+1)] = -1
                    v[(off_b+1)*5 + (off_c+0)] = 1
                    v[(off_b+1)*5 + (off_c+1)] = 1
                    w[(off_c+0)*5 + (off_a+0)] = 1
                    add_col(u, v, w)

                else:
                    # Schoolbook
                    for u_idx in range(size_a):
                        for v_idx in range(size_b):
                            for w_idx in range(size_c):
                                u, v, w = zeros()
                                gi = off_a + u_idx
                                gj = off_b + v_idx
                                gk = off_c + w_idx
                                
                                u[gi*5 + gj] = 1
                                v[gj*5 + gk] = 1
                                w[gk*5 + gi] = 1
                                add_col(u, v, w)
                                
    # Convert lists of columns to lists of rows
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

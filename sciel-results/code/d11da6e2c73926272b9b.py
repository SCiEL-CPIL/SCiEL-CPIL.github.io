"""Rewritten program for Matrix multiplication 3,3,4 using Block Strassen decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9xR, V: 12xR, W: 12xR
    
    # Rank 34 decomposition
    R = 34
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    def add_component(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Helper to map A index (i,j) to U row: i*m + j
    def u_row(i, j): return i * m + j
    # Helper to map B index (j,k) to V row: j*p + k
    def v_row(j, k): return j * p + k
    # Helper to map C index (i,k) to W row: k*n + i (Trace convention W[k*n+i, r])
    def w_row(i, k): return k * n + i

    col = 0
    
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:4]
    # Split B into B1 (cols 0,1) and B2 (cols 2,3)
    
    # Block 1: Cols 0, 1
    b_k1, b_k2 = 0, 1
    
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    add_component(u_row(0,0), 1, v_row(0,b_k1), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(1,b_k2), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(0,0), 1, v_row(0,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(1,1), 1, v_row(1,b_k2), 1, w_row(1, b_k2), 1, col)
    col += 1
    
    # M2 = (A10 + A11)B00 -> C10
    add_component(u_row(1,0), 1, v_row(0,b_k1), 1, w_row(1, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(0,b_k1), 1, w_row(1, b_k1), 1, col)
    col += 1
    
    # M3 = A00(B01 - B11) -> C01
    add_component(u_row(0,0), 1, v_row(0,b_k2), 1, w_row(0, b_k2), 1, col)
    add_component(u_row(0,0), 1, v_row(1,b_k2), -1, w_row(0, b_k2), 1, col)
    col += 1
    
    # M4 = A11(B10 - B00) -> C00, C10
    add_component(u_row(1,1), 1, v_row(1,b_k1), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(0,b_k1), -1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(1,b_k1), 1, w_row(1, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(0,b_k1), -1, w_row(1, b_k1), 1, col)
    col += 1
    
    # M5 = (A00 + A01)B11 -> C01, C00
    add_component(u_row(0,0), 1, v_row(1,b_k2), 1, w_row(0, b_k2), 1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k2), 1, w_row(0, b_k2), 1, col)
    add_component(u_row(0,0), 1, v_row(1,b_k2), 1, w_row(0, b_k1), -1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k2), 1, w_row(0, b_k1), -1, col)
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    add_component(u_row(1,0), 1, v_row(0,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(0,0), -1, v_row(0,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(1,0), 1, v_row(0,b_k2), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(0,0), -1, v_row(0,b_k2), 1, w_row(1, b_k2), 1, col)
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11) -> C00, C11
    add_component(u_row(0,1), 1, v_row(1,b_k1), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), -1, v_row(1,b_k1), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k2), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), -1, v_row(1,b_k2), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(1,1), -1, v_row(1,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k2), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(1,1), -1, v_row(1,b_k2), 1, w_row(1, b_k2), 1, col)
    col += 1
    
    # Block 2: Cols 2, 3
    b_k1, b_k2 = 2, 3
    
    # M1 = (A00 + A11)(B02 + B13) -> C02, C13
    add_component(u_row(0,0), 1, v_row(0,b_k1), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(1,b_k2), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(0,0), 1, v_row(0,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(1,1), 1, v_row(1,b_k2), 1, w_row(1, b_k2), 1, col)
    col += 1
    
    # M2 = (A10 + A11)B02 -> C12
    add_component(u_row(1,0), 1, v_row(0,b_k1), 1, w_row(1, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(0,b_k1), 1, w_row(1, b_k1), 1, col)
    col += 1
    
    # M3 = A00(B03 - B13) -> C03
    add_component(u_row(0,0), 1, v_row(0,b_k2), 1, w_row(0, b_k2), 1, col)
    add_component(u_row(0,0), 1, v_row(1,b_k2), -1, w_row(0, b_k2), 1, col)
    col += 1
    
    # M4 = A11(B12 - B02) -> C02, C12
    add_component(u_row(1,1), 1, v_row(1,b_k1), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(0,b_k1), -1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(1,b_k1), 1, w_row(1, b_k1), 1, col)
    add_component(u_row(1,1), 1, v_row(0,b_k1), -1, w_row(1, b_k1), 1, col)
    col += 1
    
    # M5 = (A00 + A01)B13 -> C03, C02
    add_component(u_row(0,0), 1, v_row(1,b_k2), 1, w_row(0, b_k2), 1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k2), 1, w_row(0, b_k2), 1, col)
    add_component(u_row(0,0), 1, v_row(1,b_k2), 1, w_row(0, b_k1), -1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k2), 1, w_row(0, b_k1), -1, col)
    col += 1
    
    # M6 = (A10 - A00)(B02 + B03) -> C13
    add_component(u_row(1,0), 1, v_row(0,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(0,0), -1, v_row(0,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(1,0), 1, v_row(0,b_k2), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(0,0), -1, v_row(0,b_k2), 1, w_row(1, b_k2), 1, col)
    col += 1
    
    # M7 = (A01 - A11)(B12 + B13) -> C02, C13
    add_component(u_row(0,1), 1, v_row(1,b_k1), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), -1, v_row(1,b_k1), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k2), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(1,1), -1, v_row(1,b_k2), 1, w_row(0, b_k1), 1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(1,1), -1, v_row(1,b_k1), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(0,1), 1, v_row(1,b_k2), 1, w_row(1, b_k2), 1, col)
    add_component(u_row(1,1), -1, v_row(1,b_k2), 1, w_row(1, b_k2), 1, col)
    col += 1
    
    # 2. Outer product u w^T (8 components)
    # u = A[0,2], A[1,2]
    # w = B[2,0..3]
    for i in range(2):
        for k in range(4):
            add_component(u_row(i, 2), 1, v_row(2, k), 1, w_row(i, k), 1, col)
            col += 1
            
    # 3. Dot product v^T B24 (8 components)
    # v = A[2,0], A[2,1]
    # B24 rows 0,1 cols 0..3
    for j in range(2):
        for k in range(4):
            add_component(u_row(2, j), 1, v_row(j, k), 1, w_row(2, k), 1, col)
            col += 1
            
    # 4. Scalar vector product alpha w^T (4 components)
    # alpha = A[2,2]
    # w = B[2,0..3]
    for k in range(4):
        add_component(u_row(2, 2), 1, v_row(2, k), 1, w_row(2, k), 1, col)
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

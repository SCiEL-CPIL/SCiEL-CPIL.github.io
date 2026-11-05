"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Rank estimate: 4 * (14 + 11) = 100
    rank = 100
    
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 100
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 100
    W = [[0.0] * rank for _ in range(p * n)]  # 28 x 100
    
    col = 0
    
    # Helper to add a local term
    # a_idx, b_idx, c_idx are the global indices for the specific element
    # val_u, val_v, val_w are the coefficients
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Helper to add Strassen 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # a_row, a_col: top-left index of A block
    # b_row, b_col: top-left index of B block
    # c_row, c_col: top-left index of C block
    def add_strassen_2x2x2(a_row, a_col, b_row, b_col, c_row, c_col):
        # A indices: (a_row+a_r, a_col+a_c)
        # B indices: (b_row+b_r, b_col+b_c)
        # C indices: (c_row+c_r, c_col+c_c)
        
        # m1 = (a+d)(e+h)
        # A part: a + d -> indices (0,0) and (1,1)
        # B part: e + h -> indices (0,0) and (1,1)
        # C part: p + s - q - r? No.
        # p = m1 + m4 - m5 + m7
        # q = m3 + m5
        # r = m2 + m4
        # s = m1 - m2 + m3 + m6
        
        # We need to map the coefficients to the global indices.
        
        # m1
        # A: (1, 0, 0, 1) -> flat: [1, 0, 0, 1]
        # B: (1, 0, 0, 1) -> flat: [1, 0, 0, 1]
        # C: (1, 0, 0, 1) -> flat: [1, 0, 0, 1] (p=1, q=0, r=0, s=1)
        
        # This is getting complex to hardcode the flat indices.
        # Let's create local vectors and map them.
        
        def get_local_indices(a_r, a_c, b_r, b_c, c_r, c_c):
            # Returns global indices for local (r, c)
            u_idx = (a_r + r) * m + (a_c + c)
            v_idx = (b_r + r) * p + (b_c + c)
            w_idx = (c_r + r) * n + (c_c + c)
            return u_idx, v_idx, w_idx

        # m1: (a+d)(e+h)
        # A coeffs: a=1, d=1. Others 0.
        # B coeffs: e=1, h=1. Others 0.
        # C coeffs: p=1, s=1. Others 0.
        # p is (0,0) of C, s is (1,1) of C.
        
        # We add one term for the product of the linear forms.
        # U vector: non-zero at (0,0) and (1,1) of A block.
        # V vector: non-zero at (0,0) and (1,1) of B block.
        # W vector: non-zero at (0,0) and (1,1) of C block.
        
        # But wait, the term is $m_1$.
        # $C_{0,0} += 1 \cdot m_1$
        # $C_{1,1} += 1 \cdot m_1$
        # So W has 1 at (0,0) and 1 at (1,1).
        
        # U:
        U[(a_row + 0) * m + (a_col + 0)][col] = 1
        U[(a_row + 1) * m + (a_col + 1)][col] = 1
        # V:
        V[(b_row + 0) * p + (b_col + 0)][col] = 1
        V[(b_row + 1) * p + (b_col + 1)][col] = 1
        # W:
        W[(c_row + 0) * n + (c_col + 0)][col] = 1
        W[(c_row + 1) * n + (c_col + 1)][col] = 1
        col += 1
        
        # m2: (c+d)e
        # A: c=1, d=1. (1,0) and (1,1)
        # B: e=1. (0,0)
        # C: r=1. (1,0)
        U[(a_row + 1) * m + (a_col + 0)][col] = 1
        U[(a_row + 1) * m + (a_col + 1)][col] = 1
        V[(b_row + 0) * p + (b_col + 0)][col] = 1
        W[(c_row + 1) * n + (c_col + 0)][col] = 1
        col += 1
        
        # m3: a(f-h)
        # A: a=1. (0,0)
        # B: f=1, h=-1. (0,1) and (1,1)
        # C: q=1. (0,1)
        U[(a_row + 0) * m + (a_col + 0)][col] = 1
        V[(b_row + 0) * p + (b_col + 1)][col] = 1
        V[(b_row + 1) * p + (b_col + 1)][col] = -1
        W[(c_row + 0) * n + (c_col + 1)][col] = 1
        col += 1
        
        # m4: d(g-e)
        # A: d=1. (1,1)
        # B: g=1, e=-1. (1,0) and (0,0)
        # C: p=1, r=1. (0,0) and (1,0)
        U[(a_row + 1) * m + (a_col + 1)][col] = 1
        V[(b_row + 1) * p + (b_col + 0)][col] = 1
        V[(b_row + 0) * p + (b_col + 0)][col] = -1
        W[(c_row + 0) * n + (c_col + 0)][col] = 1
        W[(c_row + 1) * n + (c_col + 0)][col] = 1
        col += 1
        
        # m5: (a+b)h
        # A: a=1, b=1. (0,0) and (0,1)
        # B: h=1. (1,1)
        # C: p=-1, q=1. (0,0) and (0,1)
        U[(a_row + 0) * m + (a_col + 0)][col] = 1
        U[(a_row + 0) * m + (a_col + 1)][col] = 1
        V[(b_row + 1) * p + (b_col + 1)][col] = 1
        W[(c_row + 0) * n + (c_col + 0)][col] = -1
        W[(c_row + 0) * n + (c_col + 1)][col] = 1
        col += 1
        
        # m6: (c-a)(e+f)
        # A: c=1, a=-1. (1,0) and (0,0)
        # B: e=1, f=1. (0,0) and (0,1)
        # C: s=1. (1,1)
        U[(a_row + 1) * m + (a_col + 0)][col] = 1
        U[(a_row + 0) * m + (a_col + 0)][col] = -1
        V[(b_row + 0) * p + (b_col + 0)][col] = 1
        V[(b_row + 0) * p + (b_col + 1)][col] = 1
        W[(c_row + 1) * n + (c_col + 1)][col] = 1
        col += 1
        
        # m7: (b-d)(g+h)
        # A: b=1, d=-1. (0,1) and (1,1)
        # B: g=1, h=1. (1,0) and (1,1)
        # C: p=1. (0,0)
        U[(a_row + 0) * m + (a_col + 1)][col] = 1
        U[(a_row + 1) * m + (a_col + 1)][col] = -1
        V[(b_row + 1) * p + (b_col + 0)][col] = 1
        V[(b_row + 1) * p + (b_col + 1)][col] = 1
        W[(c_row + 0) * n + (c_col + 0)][col] = 1
        col += 1

    # Helper to add Schoolbook 2x2x1
    # A is 2x2, B is 2x1, C is 2x1
    def add_schoolbook_2x2x1(a_row, a_col, b_row, b_col, c_row, c_col):
        # C_0 = a*e + b*g
        # C_1 = c*e + d*g
        
        # Term 1: a*e -> C_0
        U[(a_row + 0) * m + (a_col + 0)][col] = 1
        V[(b_row + 0) * p + (b_col + 0)][col] = 1
        W[(c_row + 0) * n + (c_col + 0)][col] = 1
        col += 1
        
        # Term 2: b*g -> C_0
        U[(a_row + 0) * m + (a_col + 1)][col] = 1
        V[(b_row + 1) * p + (b_col + 0)][col] = 1
        W[(c_row + 0) * n + (c_col + 0)][col] = 1
        col += 1
        
        # Term 3: c*e -> C_1
        U[(a_row + 1) * m + (a_col + 0)][col] = 1
        V[(b_row + 0) * p + (b_col + 0)][col] = 1
        W[(c_row + 1) * n + (c_col + 0)][col] = 1
        col += 1
        
        # Term 4: d*g -> C_1
        U[(a_row + 1) * m + (a_col + 1)][col] = 1
        V[(b_row + 1) * p + (b_col + 0)][col] = 1
        W[(c_row + 1) * n + (c_col + 0)][col] = 1
        col += 1

    # Block Decomposition
    # A blocks: (0,0), (0,1), (1,0), (1,1) -> 2x2
    # B blocks: (0,0) 2x4, (0,1) 2x3, (1,0) 2x4, (1,1) 2x3
    # C blocks: (0,0) 2x4, (0,1) 2x3, (1,0) 2x4, (1,1) 2x3
    
    # C_00 (2x4) = A_00 B_00 + A_01 B_10
    # A_00 B_00: A(0,0), B(0,0) 2x4. Split B into 2x2 (0,0) and 2x2 (0,2)
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0) # A_00, B_00_left, C_00_left
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2) # A_00, B_00_right, C_00_right
    
    # A_01 B_10: A(0,1), B(1,0) 2x4. Split B into 2x2 (1,0) and 2x2 (1,2)
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0) # A_01, B_10_left, C_00_left
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2) # A_01, B_10_right, C_00_right
    
    # C_01 (2x3) = A_00 B_01 + A_01 B_11
    # A_00 B_01: A(0,0), B(0,1) 2x3. Split B into 2x2 (0,4) and 2x1 (0,6)
    add_strassen_2x2x2(0, 0, 0, 4, 0, 4) # A_00, B_01_left, C_01_left
    add_schoolbook_2x2x1(0, 0, 0, 6, 0, 6) # A_00, B_01_right, C_01_right
    
    # A_01 B_11: A(0,1), B(1,1) 2x3. Split B into 2x2 (1,4) and 2x1 (1,6)
    add_strassen_2x2x2(0, 2, 2, 4, 0, 4) # A_01, B_11_left, C_01_left
    add_schoolbook_2x2x1(0, 2, 2, 6, 0, 6) # A_01, B_11_right, C_01_right
    
    # C_10 (2x4) = A_10 B_00 + A_11 B_10
    # A_10 B_00: A(1,0), B(0,0) 2x4.
    add_strassen_2x2x2(2, 0, 0, 0, 2, 0) # A_10, B_00_left, C_10_left
    add_strassen_2x2x2(2, 0, 0, 2, 2, 2) # A_10, B_00_right, C_10_right
    
    # A_11 B_10: A(1,1), B(1,0) 2x4.
    add_strassen_2x2x2(2, 2, 2, 0, 2, 0) # A_11, B_10_left, C_10_left
    add_strassen_2x2x2(2, 2, 2, 2, 2, 2) # A_11, B_10_right, C_10_right
    
    # C_11 (2x3) = A_10 B_01 + A_11 B_11
    # A_10 B_01: A(1,0), B(0,1) 2x3.
    add_strassen_2x2x2(2, 0, 0, 4, 2, 4) # A_10, B_01_left, C_11_left
    add_schoolbook_2x2x1(2, 0, 0, 6, 2, 6) # A_10, B_01_right, C_11_right
    
    # A_11 B_11: A(1,1), B(1,1) 2x3.
    add_strassen_2x2x2(2, 2, 2, 4, 2, 4) # A_11, B_11_left, C_11_left
    add_schoolbook_2x2x1(2, 2, 2, 6, 2, 6) # A_11, B_11_right, C_11_right
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

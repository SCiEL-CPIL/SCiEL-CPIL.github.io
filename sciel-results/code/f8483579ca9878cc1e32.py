"""Rewritten program: Strassen-based decomposition for 2x4 by 4x7 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing factor matrices
    for a low-rank exact decomposition of the 2x4 by 4x7 matrix multiplication tensor.
    
    Uses Strassen's algorithm on 2x2 blocks where possible to reduce rank from 56 (schoolbook).
    For 2x4 * 4x7:
    - Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    - Split B (4x7) into B1 (2x7, rows 0-1) and B2 (2x7, rows 2-3)
    - C = A*B = A1*B1 + A2*B2
    
    For each 2x2 * 2x7 product, we can use Strassen on 2x2 blocks of B.
    B1 is 2x7, so we split into three 2x2 blocks (cols 0-1, 2-3, 4-5) and one 2x1 block (col 6).
    - 2x2 * 2x2 uses Strassen: rank 7
    - 2x2 * 2x1 uses standard: rank 4 (or could use a 2x2*2x2 Strassen with zero padding)
    
    Actually, let's think more carefully. We can treat each 2x2 * 2x7 as:
    Three 2x2 * 2x2 Strassen products (rank 7 each) = 21
    One 2x2 * 2x1 product (rank 4) = 4
    Total per A_i * B_i: 25
    Total: 50
    
    But wait, we can also use Strassen for the 2x2 * 2x1 by padding to 2x2 * 2x2.
    That would give rank 7 instead of 4, which is worse. So stick with 4 for the last column.
    
    Actually, let me reconsider. The standard schoolbook for 2x2 * 2x1 is:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    This is rank 4 (four multiplications).
    
    For Strassen on 2x2 * 2x2, we have rank 7.
    
    So for 2x2 * 2x7:
    - Cols 0-1: Strassen, rank 7
    - Cols 2-3: Strassen, rank 7
    - Cols 4-5: Strassen, rank 7
    - Col 6: standard, rank 4
    Total: 25
    
    Two such products (A1*B1 and A2*B2): total rank 50.
    
    This is better than schoolbook rank 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    U_rows = n * m  # 8
    V_rows = m * p  # 28
    W_rows = p * n  # 14
    
    factors = []  # Each factor is (u_vec, v_vec, w_vec)
    
    def make_vec(size, default=0.0):
        return [default] * size
    
    def add_factor(u_entries, v_entries, w_entries):
        """
        u_entries: dict mapping row_index_in_U -> value
        v_entries: dict mapping row_index_in_V -> value
        w_entries: dict mapping row_index_in_W -> value
        """
        u = make_vec(U_rows)
        v = make_vec(V_rows)
        w = make_vec(W_rows)
        for idx, val in u_entries.items():
            u[idx] = val
        for idx, val in v_entries.items():
            v[idx] = val
        for idx, val in w_entries.items():
            w[idx] = val
        factors.append((u, v, w))
    
    # Strassen factors for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)h
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)e
    # P6 = (c+a)(f-e)
    # P7 = (b+d)(g-f)
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    
    def strassen_2x2_2x2(a_offset, b_offset, c_offset, d_offset, 
                          e_offset, f_offset, g_offset, h_offset,
                          co_col0, co_col1, row0, row1, col_base):
        """
        Add 7 Strassen factors for multiplying a 2x2 A submatrix by a 2x2 B submatrix.
        
        A is 2x2, with elements at:
          a = A[row0, col_base]
          b = A[row0, col_base+1]
          c = A[row1, col_base]
          d = A[row1, col_base+1]
        
        B is 2x2, with elements at:
          e = B[e_offset, 0]  (relative to B's local coords)
          f = B[e_offset, 1]
          g = B[f_offset, 0]
          h = B[f_offset, 1]
        
        Wait, I need to be careful with the indexing.
        
        For A1 (rows 0-1 of A, cols 0-1 of A):
          a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
        For A2 (rows 0-1 of A, cols 2-3 of A):
          a = A[0,2], b = A[0,3], c = A[1,2], d = A[1,3]
        
        For B1 (rows 0-1 of B):
          For cols 0-1: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
          For cols 2-3: e=B[0,2], f=B[0,3], g=B[1,2], h=B[1,3]
          For cols 4-5: e=B[0,4], f=B[0,5], g=B[1,4], h=B[1,5]
        
        For B2 (rows 2-3 of B):
          For cols 0-1: e=B[2,0], f=B[2,1], g=B[3,0], h=B[3,1]
          For cols 2-3: e=B[2,2], f=B[2,3], g=B[3,2], h=B[3,3]
          For cols 4-5: e=B[2,4], f=B[2,5], g=B[3,4], h=B[3,5]
        
        U index for A[i,j] is i*m + j = i*4 + j
        V index for B[i,j] is i*p + j = i*7 + j
        W index for C[i,j] is j*n + i = j*2 + i (official trace convention W[k*n+i,r])
        """
        # A elements
        a_idx = a_offset  # U index
        b_idx = b_offset
        c_idx = c_offset
        d_idx = d_offset
        
        # B elements
        e_idx = e_offset  # V index
        f_idx = f_offset
        g_idx = g_offset
        h_idx = h_offset
        
        # C elements
        c00_idx = co_col0  # W index for C[row0, col0]
        c01_idx = co_col1  # W index for C[row0, col1]
        c10_idx = co_col0 + 1  # Wait, W index is j*n+i, so C[row1, col0] and C[row1, col1]
        c11_idx = co_col1 + 1
        
        # P1 = (a+d)(e+h)
        # U: a=1, d=1; V: e=1, h=1; W: c00=1, c11=1
        add_factor(
            {a_idx: 1, d_idx: 1},
            {e_idx: 1, h_idx: 1},
            {c00_idx: 1, c11_idx: 1}
        )
        
        # P2 = (c+d)h
        # U: c=1, d=1; V: h=1; W: c10=1, c11=-1
        add_factor(
            {c_idx: 1, d_idx: 1},
            {h_idx: 1},
            {c10_idx: 1, c11_idx: -1}
        )
        
        # P3 = a(f-h)
        # U: a=1; V: f=1, h=-1; W: c01=1, c11=1
        add_factor(
            {a_idx: 1},
            {f_idx: 1, h_idx: -1},
            {c01_idx: 1, c11_idx: 1}
        )
        
        # P4 = d(g-e)
        # U: d=1; V: g=1, e=-1; W: c00=1, c10=1
        add_factor(
            {d_idx: 1},
            {g_idx: 1, e_idx: -1},
            {c00_idx: 1, c10_idx: 1}
        )
        
        # P5 = (a+b)e
        # U: a=1, b=1; V: e=1; W: c00=-1, c01=1
        add_factor(
            {a_idx: 1, b_idx: 1},
            {e_idx: 1},
            {c00_idx: -1, c01_idx: 1}
        )
        
        # P6 = (c+a)(f-e)
        # U: c=1, a=1; V: f=1, e=-1; W: c11=1
        add_factor(
            {c_idx: 1, a_idx: 1},
            {f_idx: 1, e_idx: -1},
            {c11_idx: 1}
        )
        
        # P7 = (b+d)(g-f)
        # U: b=1, d=1; V: g=1, f=-1; W: c00=1
        add_factor(
            {b_idx: 1, d_idx: 1},
            {g_idx: 1, f_idx: -1},
            {c00_idx: 1}
        )
    
    def standard_2x2_2x1(a_offset, b_offset, c_offset, d_offset,
                          e_offset, g_offset,
                          col0, row0, row1):
        """
        Add 4 standard factors for multiplying a 2x2 A submatrix by a 2x1 B submatrix.
        
        A = [[a, b], [c, d]]
        B = [[e], [g]]
        C[0,0] = a*e + b*g
        C[1,0] = c*e + d*g
        
        W index for C[row, col0] is col0*n + row
        """
        c0_idx = col0 * n + row0  # W index for C[row0, col0]
        c1_idx = col0 * n + row1  # W index for C[row1, col0]
        
        # Q1 = a*e
        add_factor(
            {a_offset: 1},
            {e_offset: 1},
            {c0_idx: 1}
        )
        
        # Q2 = b*g
        add_factor(
            {b_offset: 1},
            {g_offset: 1},
            {c0_idx: 1}
        )
        
        # Q3 = c*e
        add_factor(
            {c_offset: 1},
            {e_offset: 1},
            {c1_idx: 1}
        )
        
        # Q4 = d*g
        add_factor(
            {d_offset: 1},
            {g_offset: 1},
            {c1_idx: 1}
        )
    
    # Process A1 * B1
    # A1: rows 0-1, cols 0-1 of A
    # B1: rows 0-1 of B, all 7 cols
    
    # A1 elements (U indices):
    # A[0,0] -> 0*4+0 = 0
    # A[0,1] -> 0*4+1 = 1
    # A[1,0] -> 1*4+0 = 4
    # A[1,1] -> 1*4+1 = 5
    a1_a, a1_b, a1_c, a1_d = 0, 1, 4, 5
    
    # B1 elements (V indices): B[row, col] -> row*7 + col
    # For cols 0-1:
    b1_e0, b1_f0, b1_g0, b1_h0 = 0*7+0, 0*7+1, 1*7+0, 1*7+1
    # For cols 2-3:
    b1_e1, b1_f1, b1_g1, b1_h1 = 0*7+2, 0*7+3, 1*7+2, 1*7+3
    # For cols 4-5:
    b1_e2, b1_f2, b1_g2, b1_h2 = 0*7+4, 0*7+5, 1*7+4, 1*7+5
    # For col 6:
    b1_e3, b1_g3 = 0*7+6, 1*7+6
    
    # C elements (W indices): C[row, col] -> col*2 + row
    # For cols 0-1:
    # C[0,0] -> 0*2+0 = 0, C[0,1] -> 1*2+0 = 2
    # C[1,0] -> 0*2+1 = 1, C[1,1] -> 1*2+1 = 3
    c1_00, c1_01 = 0, 2
    c1_10, c1_11 = 1, 3
    
    # For cols 2-3:
    # C[0,2] -> 2*2+0 = 4, C[0,3] -> 3*2+0 = 6
    # C[1,2] -> 2*2+1 = 5, C[1,3] -> 3*2+1 = 7
    c2_00, c2_01 = 4, 6
    c2_10, c2_11 = 5, 7
    
    # For cols 4-5:
    # C[0,4] -> 4*2+0 = 8, C[0,5] -> 5*2+0 = 10
    # C[1,4] -> 4*2+1 = 9, C[1,5] -> 5*2+1 = 11
    c3_00, c3_01 = 8, 10
    c3_10, c3_11 = 9, 11
    
    # For col 6:
    # C[0,6] -> 6*2+0 = 12
    # C[1,6] -> 6*2+1 = 13
    c4_00, c4_10 = 12, 13
    
    # Add Strassen factors for A1 * B1
    strassen_2x2_2x2(a1_a, a1_b, a1_c, a1_d,
                     b1_e0, b1_f0, b1_g0, b1_h0,
                     c1_00, c1_01, 0, 1, 0)
    strassen_2x2_2x2(a1_a, a1_b, a1_c, a1_d,
                     b1_e1, b1_f1, b1_g1, b1_h1,
                     c2_00, c2_01, 0, 1, 2)
    strassen_2x2_2x2(a1_a, a1_b, a1_c, a1_d,
                     b1_e2, b1_f2, b1_g2, b1_h2,
                     c3_00, c3_01, 0, 1, 4)
    standard_2x2_2x1(a1_a, a1_b, a1_c, a1_d,
                     b1_e3, b1_g3,
                     6, 0, 1)
    
    # Process A2 * B2
    # A2: rows 0-1, cols 2-3 of A
    # B2: rows 2-3 of B, all 7 cols
    
    # A2 elements (U indices):
    # A[0,2] -> 0*4+2 = 2
    # A[0,3] -> 0*4+3 = 3
    # A[1,2] -> 1*4+2 = 6
    # A[1,3] -> 1*4+3 = 7
    a2_a, a2_b, a2_c, a2_d = 2, 3, 6, 7
    
    # B2 elements (V indices):
    # For cols 0-1:
    b2_e0, b2_f0, b2_g0, b2_h0 = 2*7+0, 2*7+1, 3*7+0, 3*7+1
    # For cols 2-3:
    b2_e1, b2_f1, b2_g1, b2_h1 = 2*7+2, 2*7+3, 3*7+2, 3*7+3
    # For cols 4-5:
    b2_e2, b2_f2, b2_g2, b2_h2 = 2*7+4, 2*7+5, 3*7+4, 3*7+5
    # For col 6:
    b2_e3, b2_g3 = 2*7+6, 3*7+6
    
    # C elements (W indices) - same as before since C is the same
    # Reuse the c indices
    
    # Add Strassen factors for A2 * B2
    strassen_2x2_2x2(a2_a, a2_b, a2_c, a2_d,
                     b2_e0, b2_f0, b2_g0, b2_h0,
                     c1_00, c1_01, 0, 1, 0)
    strassen_2x2_2x2(a2_a, a2_b, a2_c, a2_d,
                     b2_e1, b2_f1, b2_g1, b2_h1,
                     c2_00, c2_01, 0, 1, 2)
    strassen_2x2_2x2(a2_a, a2_b, a2_c, a2_d,
                     b2_e2, b2_f2, b2_g2, b2_h2,
                     c3_00, c3_01, 0, 1, 4)
    standard_2x2_2x1(a2_a, a2_b, a2_c, a2_d,
                     b2_e3, b2_g3,
                     6, 0, 1)
    
    rank = len(factors)
    print(f"Constructed decomposition with rank {rank}")
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for col, (u, v, w) in enumerate(factors):
        for row in range(U_rows):
            if u[row] != 0:
                U[row][col] = u[row]
        for row in range(V_rows):
            if v[row] != 0:
                V[row][col] = v[row]
        for row in range(W_rows):
            if w[row] != 0:
                W[row][col] = w[row]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

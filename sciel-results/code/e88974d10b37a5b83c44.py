"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    # Block-Kronecker decomposition:
    # - First 2 columns of B (3x2 block): rank 11 (Strassen 2x2 + schoolbook 2x1)
    # - Remaining 8 columns of B (3x8 block): schoolbook rank 48
    # Total rank: 11 + 48 = 59

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_pairs, v_pairs, w_pairs):
        """Add a rank-1 term with sparse non-zero entries.
        Each list contains (index, coefficient) pairs."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, coeff in u_pairs:
            u_vec[idx] = coeff
        for idx, coeff in v_pairs:
            v_vec[idx] = coeff
        for idx, coeff in w_pairs:
            w_vec[idx] = coeff
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # --- Block 1: B1 (3x2), columns k=0,1 ---
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01], [b10, b11], [b20, b21]]
    # C1 = A*B1 (2x2)
    #
    # Partition: A1 = [[a00,a01],[a10,a11]] (2x2), A2 = [[a02],[a12]] (2x1)
    #            B11 = [[b00,b01],[b10,b11]] (2x2), B12 = [[b20,b21]] (1x2)
    #
    # C1 = A1*B11 + A2*B12
    #
    # Strassen for A1*B11 (rank 7):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6
    #
    # Plus A2*B12 (schoolbook, rank 4):
    # C11 += a02*b20
    # C12 += a02*b21
    # C21 += a12*b20
    # C22 += a12*b21
    #
    # Total for Block 1: 7 + 4 = 11 terms

    # Index conventions:
    # U: n*m = 6 rows, index i*m+j (i=row of A, j=col of A / row of B)
    #   a00: idx 0, a01: idx 1, a02: idx 2, a10: idx 3, a11: idx 4, a12: idx 5
    # V: m*p = 30 rows, index j*p+k (j=row of B, k=col of B)
    #   b00: idx 0, b01: idx 1, b10: idx 10, b11: idx 11, b20: idx 20, b21: idx 21
    # W: p*n = 20 rows, index k*n+i (k=col of C, i=row of C)
    #   C00: idx 0, C01: idx 1, C10: idx 2, C11: idx 3

    # M1 = (a00+a11)*(b00+b11)
    # U: a00 (idx 0, coeff 1), a11 (idx 4, coeff 1)
    # V: b00 (idx 0, coeff 1), b11 (idx 11, coeff 1)
    # W: C00 (idx 0, coeff 1), C11 (idx 3, coeff 1)
    add_term([(0, 1), (4, 1)], [(0, 1), (11, 1)], [(0, 1), (3, 1)])

    # M2 = (a10+a11)*b00
    # U: a10 (idx 3, coeff 1), a11 (idx 4, coeff 1)
    # V: b00 (idx 0, coeff 1)
    # W: C10 (idx 2, coeff 1), C11 (idx 3, coeff -1)  [C10 += M2, C11 -= M2]
    add_term([(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)])

    # M3 = a00*(b01-b11)
    # U: a00 (idx 0, coeff 1)
    # V: b01 (idx 1, coeff 1), b11 (idx 11, coeff -1)
    # W: C01 (idx 1, coeff 1)  [C01 += M3]
    add_term([(0, 1)], [(1, 1), (11, -1)], [(1, 1)])

    # M4 = a11*(b10-b00)
    # U: a11 (idx 4, coeff 1)
    # V: b10 (idx 10, coeff 1), b00 (idx 0, coeff -1)
    # W: C10 (idx 2, coeff 1)  [C10 += M4]
    add_term([(4, 1)], [(10, 1), (0, -1)], [(2, 1)])

    # M5 = (a00+a01)*b11
    # U: a00 (idx 0, coeff 1), a01 (idx 1, coeff 1)
    # V: b11 (idx 11, coeff 1)
    # W: C00 (idx 0, coeff -1), C01 (idx 1, coeff 1)  [C00 -= M5, C01 += M5]
    add_term([(0, 1), (1, 1)], [(11, 1)], [(0, -1), (1, 1)])

    # M6 = (a10-a00)*(b00+b01)
    # U: a10 (idx 3, coeff 1), a00 (idx 0, coeff -1)
    # V: b00 (idx 0, coeff 1), b01 (idx 1, coeff 1)
    # W: C11 (idx 3, coeff 1)  [C11 += M6]
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])

    # M7 = (a01-a11)*(b10+b11)
    # U: a01 (idx 1, coeff 1), a11 (idx 4, coeff -1)
    # V: b10 (idx 10, coeff 1), b11 (idx 11, coeff 1)
    # W: C00 (idx 0, coeff 1)  [C00 += M7]
    add_term([(1, 1), (4, -1)], [(10, 1), (11, 1)], [(0, 1)])

    # Schoolbook for A2*B12 (rank 4):
    # a02*b20: U idx 2, V idx 20, W idx 0
    add_term([(2, 1)], [(20, 1)], [(0, 1)])
    # a02*b21: U idx 2, V idx 21, W idx 2
    add_term([(2, 1)], [(21, 1)], [(2, 1)])
    # a12*b20: U idx 5, V idx 20, W idx 1
    add_term([(5, 1)], [(20, 1)], [(1, 1)])
    # a12*b21: U idx 5, V idx 21, W idx 3
    add_term([(5, 1)], [(21, 1)], [(3, 1)])

    # --- Block 2: B2 (3x8), columns k=2..9 ---
    # Schoolbook for remaining 8 columns: rank 2*3*8 = 48
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

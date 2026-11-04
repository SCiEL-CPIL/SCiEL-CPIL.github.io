"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8

    U = []
    V = []
    W = []

    def add_term(u_entries, v_entries, w_entries):
        """
        Add a rank-1 term where u_entries, v_entries, w_entries are lists of (index, coeff).
        """
        u_row = [0] * (n * m)
        for idx, coeff in u_entries:
            u_row[idx] = coeff
        v_row = [0] * (m * p)
        for idx, coeff in v_entries:
            v_row[idx] = coeff
        w_row = [0] * (p * n)
        for idx, coeff in w_entries:
            w_row[idx] = coeff
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)

    # Strassen 2x2x2 decomposition terms
    # Local indices:
    # A: 00->0, 01->1, 10->2, 11->3
    # B: 00->0, 01->1, 10->2, 11->3
    # C: 00->0, 01->1, 10->2, 11->3
    strassen_terms = [
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        ([(2,1), (3,1)], [(0,1)],           [(2,1), (3,1)]),
        ([(0,1)],           [(1,1), (3,-1)], [(1,1)]),
        ([(3,1)],           [(2,1), (0,-1)], [(2,1)]),
        ([(0,1), (2,1)], [(2,1)],           [(0,-1), (2,1)]),
        ([(1,1), (0,-1)], [(0,1), (2,1)],   [(1,1), (3,-1)]),
        ([(3,1), (2,-1)], [(1,1), (3,1)],   [(1,1), (0,1)]),
    ]

    # We have 4 column pairs: (0,1), (2,3), (4,5), (6,7)
    column_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

    for k0, k1 in column_pairs:
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1+1] -> C[0:2, k0:k1+1]
        a_map = {0: 0, 1: 1, 2: 3, 3: 4}
        b_map = {0: k0, 1: k1, 2: 8 + k0, 3: 8 + k1}
        c_map = {0: k0 * 3, 1: k0 * 3 + 1, 2: k1 * 3, 3: k1 * 3 + 1}

        for u, v, w in strassen_terms:
            u_list = [(a_map[i], u[i]) for i in range(4) if u[i] != 0]
            v_list = [(b_map[i], v[i]) for i in range(4) if v[i] != 0]
            w_list = [(c_map[i], w[i]) for i in range(4) if w[i] != 0]
            add_term(u_list, v_list, w_list)

        # Row 2 of A (i=2), cols 0,1 of B (j=0,1)
        # C[2, k] += A[2,0]B[0,k] + A[2,1]B[1,k]
        for k in [k0, k1]:
            add_term([(6, 1)], [(k, 1)], [(k * 3 + 2, 1)])
            add_term([(7, 1)], [(8 + k, 1)], [(k * 3 + 2, 1)])

        # Col 2 of A (j=2), rows 0,1 (i=0,1)
        # C[i, k] += A[i,2]B[2,k]
        for k in [k0, k1]:
            add_term([(2, 1)], [(16 + k, 1)], [(k * 3, 1)])
            add_term([(5, 1)], [(16 + k, 1)], [(k * 3 + 1, 1)])

    # Row 2 of A (i=2), col 2 of B (j=2)
    # C[2, k] += A[2,2]B[2,k]
    for k in range(p):
        add_term([(8, 1)], [(16 + k, 1)], [(k * 3 + 2, 1)])

    # Col 2 of A (j=2), row 2 of B (i=2) is already handled above? 
    # Wait, C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2] + A[2,2]B[2,2]
    # The first two are handled in the "Row 2 of A" block for k=2.
    # The third is A[2,2]B[2,2] which is handled in the last block.
    # What about C[i, k] for i=0,1 and j=2? Handled in "Col 2 of A" block.
    # What about C[2, k] for i=2 and j=0,1? Handled in "Row 2 of A" block.
    # What about C[2, k] for i=2 and j=2? Handled in last block.
    
    # Let's double check coverage.
    # C[i, k] = sum_j A[i,j] B[j,k]
    # i=0,1:
    #   j=0,1: Covered by Strassen for each (k0, k1) pair.
    #   j=2: Covered by "Col 2 of A" block for each k.
    # i=2:
    #   j=0,1: Covered by "Row 2 of A" block for each (k0, k1) pair.
    #   j=2: Covered by last block for each k.
    
    # Total rank:
    # 4 pairs * 7 (Strassen) = 28
    # 4 pairs * 4 (Row 2, j=0,1) = 16
    # 4 pairs * 2 (Col 2, i=0,1) = 8
    # 1 block * 8 (Row 2, j=2) = 8
    # Total = 28 + 16 + 8 + 8 = 60.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

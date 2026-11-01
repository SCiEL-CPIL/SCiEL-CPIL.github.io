"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x2 * 2x10 multiplication into five independent 2x2 * 2x2 blocks.
    Each 2x2 * 2x2 block uses the rank-7 Strassen decomposition.
    Total rank = 5 * 7 = 35.

    Indexing conventions:
    - A is n x m = 2 x 2, indexed as A[i, j]
    - B is m x p = 2 x 10, indexed as B[j, k]
    - C is n x p = 2 x 10, indexed as C[i, k]

    U has shape (n*m, rank) = (4, 35)
    V has shape (m*p, rank) = (20, 35)
    W has shape (p*n, rank) = (20, 35) with W[k*n + i, c]

    For each block r (0..4) and each Strassen term t (0..6):
      U[i*m + j, r*7 + t] = U_strassen[i, j, t]
      V[j*p + (2*r + k), r*7 + t] = V_strassen[j, k, t]
      W[(2*r + k)*n + i, r*7 + t] = W_strassen[k, i, t]
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks

    # Strassen rank-7 decomposition for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B
    #
    # Strassen's 7 products:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # Verification:
    # C[0,0] = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    #        = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    #        = ae + bg  ✓
    #
    # C[0,1] = a(f-h) + (a+b)h
    #        = af - ah + ah + bh
    #        = af + bh  ✓
    #
    # C[1,0] = (c+d)e + d(g-e)
    #        = ce + de + dg - de
    #        = ce + dg  ✓
    #
    # C[1,1] = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    #        = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
    #        = dh + cf  ✓

    # Define the 7 Strassen terms as (U_entries, V_entries, W_entries)
    # U_entries: dict from (i,j) to value
    # V_entries: dict from (j,k) to value
    # W_entries: dict from (k,i) to value

    strassen_terms = [
        # Term 0: M1 = (a+d)(e+h)
        (
            {(0,0): 1, (1,1): 1},   # U: a+d
            {(0,0): 1, (1,1): 1},   # V: e+h
            {(0,0): 1, (1,1): 1}    # W: C[0,0]+=1, C[1,1]+=1
        ),
        # Term 1: M2 = (c+d)e
        (
            {(1,0): 1, (1,1): 1},   # U: c+d
            {(0,0): 1},             # V: e
            {(0,1): 1, (1,1): -1}   # W: C[1,0]+=1, C[1,1]-=1
        ),
        # Term 2: M3 = a(f-h)
        (
            {(0,0): 1},             # U: a
            {(0,1): 1, (1,1): -1},  # V: f-h
            {(1,0): 1, (1,1): 1}    # W: C[0,1]+=1, C[1,1]+=1
        ),
        # Term 3: M4 = d(g-e)
        (
            {(1,1): 1},             # U: d
            {(1,0): 1, (0,0): -1},  # V: g-e
            {(0,0): 1, (0,1): 1}    # W: C[0,0]+=1, C[1,0]+=1
        ),
        # Term 4: M5 = (a+b)h
        (
            {(0,0): 1, (0,1): 1},   # U: a+b
            {(1,1): 1},             # V: h
            {(0,0): -1, (1,0): 1}   # W: C[0,0]-=1, C[0,1]+=1
        ),
        # Term 5: M6 = (c-a)(e+f)
        (
            {(1,0): 1, (0,0): -1},  # U: c-a
            {(0,0): 1, (0,1): 1},   # V: e+f
            {(1,1): 1}              # W: C[1,1]+=1
        ),
        # Term 6: M7 = (b-d)(g+h)
        (
            {(0,1): 1, (1,1): -1},  # U: b-d
            {(1,0): 1, (1,1): 1},   # V: g+h
            {(0,0): 1}              # W: C[0,0]+=1
        ),
    ]

    rank = num_blocks * 7  # 35

    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35

    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]

            for (i, j), val in u_entries.items():
                U[i * m + j][col] = val

            for (j, k), val in v_entries.items():
                # B's column for this block is 2*r + k
                V[j * p + (2 * r + k)][col] = val

            for (k, i), val in w_entries.items():
                # C's column for this block is 2*r + k
                W[(2 * r + k) * n + i][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

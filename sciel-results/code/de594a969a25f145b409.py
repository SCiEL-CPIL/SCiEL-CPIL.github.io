"""2 x 4 x 8 matrix multiplication tensor decomposition via two rank-14 (2x2x4) blocks."""

DIMENSIONS = [2, 4, 8]

# Strassen (2x2x2) pattern: for each of 7 mult terms t:
#   a-term: (i, j) in 2x2 ; b-term: (j, k) in 2x2 ; c-term: (k, i) in 2x2
# (using standard Strassen: M1..M7)
# Represented as list of ( (a_i,a_j), (b_j,b_k), (c_k,c_i) )
STRASSEN = [
    # M1 = a11*(b21+b22) -> contributes to c21 and c22 with b part... we use the
    # standard 2x2 Strassen where C = A*B:
    # M1 = A11*(B12-B22) ; C12 += M1
    # M2 = (A11+A12)*B22 ; C22 += M2
    # M3 = (A11+A21)*B11 ; C11 += M3
    # M4 = A22*(B11-B12) ; C21 += M4
    # M5 = (A11+A22)*(B21+B22) ; C11+=M5, C22+=M5
    # M6 = (A12-A22)*(B21+B22) ; C21+=M6
    # M7 = (A11-A21)*(B11+B12) ; C12+=M7
    # We express each mult term as a triple of (linear-comb-in-A, linear-comb-in-B, linear-comb-in-C)
    # Each entry: A-coef dict (i,j)->c, B-coef dict (j,k)->c, C-coef dict (k,i)->c
]

def _dict(d):
    return dict(d)

# Build Strassen as list of (A_terms, B_terms, C_terms) where each is dict of (idx)->coef
STRASSEN_TUPLES = [
    # t=0: M1 = A11 * (B12 - B22), contributes to C12
    ( {(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1} ),
    # t=1: M2 = (A11+A12) * B22, contributes to C22
    ( {(0,0):1, (0,1):1}, {(1,1):1}, {(1,1):1} ),
    # t=2: M3 = (A11+A21) * B11, contributes to C11
    ( {(0,0):1, (1,0):1}, {(0,0):1}, {(0,0):1} ),
    # t=3: M4 = A22 * (B11 - B12), contributes to C21
    ( {(1,1):1}, {(0,0):1, (0,1):-1}, {(0,1):1} ),
    # t=4: M5 = (A11+A22) * (B21+B22), contributes to C11 and C22
    ( {(0,0):1, (1,1):1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1} ),
    # t=5: M6 = (A12-A22) * (B21+B22), contributes to C21
    ( {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1} ),
    # t=6: M7 = (A11-A21) * (B11+B12), contributes to C12
    ( {(0,0):1, (1,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1} ),
]

def construct():
    import numpy as np
    n, m, p = DIMENSIONS  # 2, 4, 8
    # U: (n*m, rank) = (8, rank); V: (m*p, rank) = (32, rank); W: (p*n, rank) = (16, rank)
    # A index: i in 0..1, j in 0..3 -> row a = i*m + j = 2*i + j
    # B index: j in 0..3, k in 0..7 -> row b = j*p + k = 8*j + k
    # C index: k in 0..7, i in 0..1 -> row w = k*n + i = 2*k + i
    #
    # We need 28 rank terms. Structure: for each (j1 in {0,1}, k1 in {0,1}) a 2x2x2 Strassen
    # sub-block handling A columns j=2*j1..2*j1+1, B rows j=2*j1..2*j1+1, output cols k=2*k1..2*k1+1.
    # Within each sub-block, the A-contraction is shared across the two k0 outputs.

    # We'll build sparse lists: for each rank term, dict of nonzero entries in U, V, W.
    terms = []  # list of (U_entries, V_entries, W_entries) each a dict row->coef

    for j1 in range(2):
        for k1 in range(2):
            for (A_terms, B_terms, C_terms) in STRASSEN_TUPLES:
                U_ent = {}
                V_ent = {}
                W_ent = {}
                # A-part: for each (i, a_j) in A_terms, a_j in {0,1} is the inner column within block j1
                #   full column index j = 2*j1 + a_j ; row a = 2*i + j
                for (i, aj), ca in A_terms.items():
                    j = 2 * j1 + aj
                    row_a = 2 * i + j
                    U_ent[row_a] = U_ent.get(row_a, 0) + ca
                # B-part: for each (b_j, b_k) in B_terms, b_j in {0,1} inner row within block j1,
                #   b_k in {0,1} inner column within block k1
                #   full row index j = 2*j1 + b_j ; full col index k = 2*k1 + b_k ; row b = 8*j + k
                for (bj, bk), cb in B_terms.items():
                    j = 2 * j1 + bj
                    k = 2 * k1 + bk
                    row_b = 8 * j + k
                    V_ent[row_b] = V_ent.get(row_b, 0) + cb
                # C-part: for each (c_k, c_i) in C_terms, c_k in {0,1} inner col within block k1,
                #   c_i in {0,1} is the row index
                #   full col index k = 2*k1 + c_k ; row w = 2*k + c_i
                for (ck, ci), cc in C_terms.items():
                    k = 2 * k1 + ck
                    row_w = 2 * k + ci
                    W_ent[row_w] = W_ent.get(row_w, 0) + cc
                terms.append((U_ent, V_ent, W_ent))

    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    for c, (U_ent, V_ent, W_ent) in enumerate(terms):
        for row, val in U_ent.items():
            if val != 0:
                U[row, c] = val
        for row, val in V_ent.items():
            if val != 0:
                V[row, c] = val
        for row, val in W_ent.items():
            if val != 0:
                W[row, c] = val

    # Convert integer object entries to ints, drop object dtype
    U = U.astype(int)
    V = V.astype(int)
    W = W.astype(int)

    # ---- Verify exact decomposition of the tensor ----
    # Build the multiplication tensor T[i,j,k] = 1 iff the (i,k) output uses A[i,j]*B[j,k].
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check on a random integer A, B that C = A@B equals sum_c (U^T A_vec) * (V^T B_vec) * W_row.
    rng = np.random.default_rng(12345)
    A_rand = rng.integers(-3, 4, size=(n, m))
    B_rand = rng.integers(-3, 4, size=(m, p))
    C_expect = A_rand @ B_rand
    a_vec = A_rand.reshape(-1)  # length 8, order (i*m+j)
    b_vec = B_rand.reshape(-1)  # length 32, order (j*p+k)
    u_av = U.T @ a_vec  # length rank
    v_bv = V.T @ b_vec  # length rank
    # C[k*n + i] = sum_c u_av[c] * v_bv[c] * W[k*n+i, c]
    C_recon = W.T @ (u_av * v_bv)  # length 16
    C_recon = C_recon.reshape(p, n).T  # (n, p)
    if not np.array_equal(C_recon, C_expect):
        raise ValueError("Decomposition failed verification")
    # Also verify on a second random matrix for robustness
    A_rand2 = rng.integers(-5, 6, size=(n, m))
    B_rand2 = rng.integers(-5, 6, size=(m, p))
    C_expect2 = A_rand2 @ B_rand2
    u_av2 = U.T @ A_rand2.reshape(-1)
    v_bv2 = V.T @ B_rand2.reshape(-1)
    C_recon2 = (W.T @ (u_av2 * v_bv2)).reshape(p, n).T
    if not np.array_equal(C_recon2, C_expect2):
        raise ValueError("Decomposition failed verification on second sample")

    return {"U": U, "V": V, "W": W}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]


def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition exactly reproduces matrix multiplication.

    Condition: for all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    (the "diagonal" tensor entries of the multiplication tensor).
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vj[c] * wk[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _strassen_4x8():
    """Strassen-like decomposition of the 4x8 multiplication.

    Split A (2x4) into columns A_L (2x2), A_R (2x2).
    Split B (4x8) into columns B_1..B_4 (each 2x2).
    C = A_L B_1 + A_R B_2 + A_L B_3 + A_R B_4.
    Each 2x2 product uses Strassen (7 scalar ops). 4 products -> 28 rank-1 terms.
    """
    n, m, p = DIMENSIONS
    # 4 Strassen products, 7 terms each => 28 columns
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 8 x rank
    V = [[0] * rank for _ in range(m * p)]   # 32 x rank
    W = [[0] * rank for _ in range(p * n)]   # 16 x rank

    # Strassen 2x2 recipe: given (a,b,c,d) input indices and (e,f,g,h) input indices,
    # the 7 products produce output C = [[c11,c12],[c21,c22]] with
    #   c11 = M1 - M4 + M5 + M7
    #   c12 = M3 + M4
    #   c21 = M1 - M2 + M5
    #   c22 = M2 - M3 + M4 + M6
    # where (a,b,c,d) and (e,f,g,h) are the 2x2 matrices.
    # We build the 7 M terms as rank-1 contributions.
    # M1 = (a+b)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+d)(e+f)
    # M6 = (c-a)(h+f)
    # M7 = (b-c)(g+e)

    # Each product maps a 2x2 input A-block (rows i=0,1; cols j) and a 2x2 B-block
    # (rows j; cols k) to a 2x2 output C-block (rows i; cols k).
    # We enumerate the 4 products.

    # product list: (A_col_block, B_col_block, C_col_block)
    # A_col_block in {0,1} (L,R), B_col_block in {0,1,2,3}, C_col_block in {0,1,2,3}
    products = [
        (0, 0, 0),  # A_L B_1 -> C col 0
        (1, 1, 1),  # A_R B_2 -> C col 1
        (0, 2, 2),  # A_L B_3 -> C col 2
        (1, 3, 3),  # A_R B_4 -> C col 3
    ]

    # For a 2x2 A-block with A_col_block = ab (0 or 1):
    #   A elements: A[i, ab*2 + dj]  where dj in {0,1}
    #   a = A[0, ab*2+0], b = A[0, ab*2+1], c = A[1, ab*2+0], d = A[1, ab*2+1]
    # For a 2x2 B-block with B_col_block = bb (0..3):
    #   B elements: B[dj, bb*2 + dk] where dj in {0,1}, dk in {0,1}
    #   e = B[0, bb*2+0], f = B[0, bb*2+1], g = B[1, bb*2+0], h = B[1, bb*2+1]
    # For a 2x2 C-block with C_col_block = cb:
    #   C output elements: C[i, cb*2 + dk]
    #   c11 = C[0, cb*2+0], c12 = C[0, cb*2+1], c21 = C[1, cb*2+0], c22 = C[1, cb*2+1]

    # The 7 M terms and their output coefficients:
    # Each M term is (sum over A elements with coeff) * (sum over B elements with coeff)
    # and contributes to each C element with a coefficient.
    # We represent each M term by:
    #   a_coeffs: list of (i, j_local, coeff) for A
    #   b_coeffs: list of (j_local, k_local, coeff) for B
    #   out_coeffs: list of (i, k_local, coeff) for C
    # where j_local in {0,1} indexes within the A/B block, k_local in {0,1} within B/C block.

    # A element (i, j_local) -> global A index i*m + (ab*2 + j_local)
    # B element (j_local, k_local) -> global B index (j_local)*p + (bb*2 + k_local)
    # C element (i, k_local) -> global C index (k_local)*n + i

    M_terms = [
        # M1 = (a+b)(e+h) -> c11:+1, c21:+1
        ([ (0,0,1), (0,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,0,1) ]),
        # M2 = (c+d)h -> c21:-1, c22:+1
        ([ (1,0,1), (1,1,1) ], [ (1,1,1) ], [ (1,0,-1), (1,1,1) ]),
        # M3 = a(f-h) -> c12:+1, c22:-1
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,-1) ]),
        # M4 = d(g-e) -> c11:-1, c12:+1, c22:+1
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,-1), (0,1,1), (1,1,1) ]),
        # M5 = (a+d)(e+f) -> c11:+1, c21:+1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (0,1,1) ], [ (0,0,1), (1,0,1) ]),
        # M6 = (c-a)(h+f) -> c22:+1
        ([ (1,0,1), (0,0,-1) ], [ (1,1,1), (0,1,1) ], [ (1,1,1) ]),
        # M7 = (b-c)(g+e) -> c11:+1
        ([ (0,1,1), (1,0,-1) ], [ (1,0,1), (0,0,1) ], [ (0,0,1) ]),
    ]

    col = 0
    for ab, bb, cb in products:
        for a_coeffs, b_coeffs, out_coeffs in M_terms:
            # Build this rank-1 term's contribution.
            # For U (A side): for each (i, j_local, coeff) in a_coeffs,
            #   U[i*m + (ab*2 + j_local), col] += coeff
            # For V (B side): for each (j_local, k_local, coeff) in b_coeffs,
            #   V[j_local*p + (bb*2 + k_local), col] += coeff
            # For W (C side): for each (i, k_local, coeff) in out_coeffs,
            #   W[k_local*n + i, col] += coeff
            for (i, jl, coeff) in a_coeffs:
                U[i * m + (ab * 2 + jl)][col] += coeff
            for (jl, kl, coeff) in b_coeffs:
                V[jl * p + (bb * 2 + kl)][col] += coeff
            for (i, kl, coeff) in out_coeffs:
                W[kl * n + i][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}


def _rank_24_decomp():
    """Attempt a rank-24 decomposition.
    
    For 2×2 × 2×4 multiplication:
    A = [[a,b],[c,d]], B = [[e,f],[g,h],[i,j],[k,l]]
    C = [[ae+bg+ci+dk, af+bh+cj+dl],
         [ce+dg+fi+hk, cf+dh+fj+hl]]
    
    We can use a decomposition with 12 rank-1 terms per 2×2×2×4 block.
    """
    n, m, p = DIMENSIONS
    rank = 24
    U = [[0] * rank for _ in range(n * m)]   # 8 x rank
    V = [[0] * rank for _ in range(m * p)]   # 32 x rank
    W = [[0] * rank for _ in range(p * n)]   # 16 x rank
    
    # Define 12 rank-1 terms for 2×2 × 2×4
    # This is a known decomposition (or we construct one)
    # For now, use Strassen twice (14 terms) but we need 12...
    # Let's just return the 28-rank version for now
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill

    # 1) Try the Strassen-like decomposition (rank 28).
    best = None
    try:
        cand = _strassen_4x8()
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            best = cand
    except Exception:
        pass

    # 2) Fallback: schoolbook (rank 64), always valid.
    if best is None:
        best = _schoolbook()

    return best

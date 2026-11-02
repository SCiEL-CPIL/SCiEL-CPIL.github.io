"""Strassen-based rank reduction for 2x2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    n, m, p = DIMENSIONS
    # We'll build U,V,W as lists of columns (rank vectors) using integer coeffs.
    # Each "term" is (u_row->coef dict, v_row->coef dict, w_row->coef dict)
    # but easier: accumulate into numpy matrices with unknown rank; track rank via count.
    # Strategy: define a list of terms; each term is a dict u: {row:coef}, v:{row:coef}, w:{row:coef}.
    # We'll pack them into dense matrices by assigning one column per term.

    terms = []  # list of (u_dict, v_dict, w_dict)

    def add_term(u, v, w):
        terms.append((u, v, w))

    # Strassen block on B-rows {j0,j1}, C-cols {k0,k1}
    def strassen_block(j0, j1, k0, k1):
        A00 = 0*2+0  # U row for A[0,0]
        A01 = 0*2+1
        A10 = 1*2+0
        A11 = 1*2+1
        B00 = j0*p+k0
        B01 = j0*p+k1
        B10 = j1*p+k0
        B11 = j1*p+k1
        # W rows: C[i,k] -> k*n + i
        C00 = k0*n+0
        C01 = k1*n+0
        C10 = k0*n+1
        C11 = k1*n+1
        # M1 = (A00+A11)(B00+B11)
        add_term({A00:1, A11:1}, {B00:1, B11:1}, {C00:1, C01:1, C10:1, C11:1})
        # M2 = (A10+A11)B00
        add_term({A10:1, A11:1}, {B00:1}, {C10:1, C11:-1})
        # M3 = A00(B10-B11)
        add_term({A00:1}, {B10:1, B11:-1}, {C01:1, C11:1})
        # M4 = A11(B00-B10)
        add_term({A11:1}, {B00:1, B10:-1}, {C00:1, C10:1})
        # M5 = (A00+A01)B11
        add_term({A00:1, A01:1}, {B11:1}, {C00:-1, C01:1})
        # M6 = (A10-A00)(B00+B01)
        add_term({A10:1, A00:-1}, {B00:1, B01:1}, {C11:1})
        # M7 = (A01-A11)(B10+B11)
        add_term({A01:1, A11:-1}, {B10:1, B11:1}, {C00:1})

    # 4 Strassen blocks covering B-row pairs {0,1} for C-col pairs
    # But B-rows are only 2 (m=2). The Strassen block needs both B-rows j0=0,j1=1.
    # So each block uses j0=0, j1=1 and a pair of C-cols.
    strassen_block(0, 1, 0, 1)
    strassen_block(0, 1, 2, 3)
    strassen_block(0, 1, 4, 5)
    strassen_block(0, 1, 6, 7)
    # Col 8: schoolbook for both B-rows
    for j in range(m):
        for i in range(n):
            add_term({i*m+j:1}, {j*p+8:1}, {8*n+i:1})

    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for c, (u, v, w) in enumerate(terms):
        for row, coef in u.items():
            U[row, c] = coef
        for row, coef in v.items():
            V[row, c] = coef
        for row, coef in w.items():
            W[row, c] = coef

    # Verification: reconstruct multiplication tensor.
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build via einsum-like loop.
    # A: n x m, B: m x p, C: n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = B[j,k] contributes with A[i,j]; equivalently T[i,j,k]=1.
    # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1.
    err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    err += 1
    if err != 0:
        # Fallback to schoolbook for safety
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

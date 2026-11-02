import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    terms = []

    def add_term(a_coeffs, b_coeffs, c_coeffs):
        """Add one rank-1 term with given coefficient dicts."""
        terms.append({'a': dict(a_coeffs), 'b': dict(b_coeffs), 'c': dict(c_coeffs)})

    def add_strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for a 2x2x2 block starting at (i0,j0,k0)."""
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i0+1,j0+1):1},
                 {(j0,k0):1, (j0+1,k0+1):1},
                 {(i0,k0):1, (i0+1,k0+1):1})
        # M2 = (A10+A11)B00
        add_term({(i0+1,j0):1, (i0+1,j0+1):1},
                 {(j0,k0):1},
                 {(i0+1,k0):1, (i0+1,k0+1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1},
                 {(j0,k0+1):1, (j0+1,k0+1):-1},
                 {(i0,k0+1):1, (i0+1,k0+1):1})
        # M4 = A11(B10-B00)
        add_term({(i0+1,j0+1):1},
                 {(j0+1,k0):1, (j0,k0):-1},
                 {(i0,k0):1, (i0+1,k0):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1, (i0,j0+1):1},
                 {(j0+1,k0+1):1},
                 {(i0,k0):-1, (i0,k0+1):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i0+1,j0):1, (i0,j0):-1},
                 {(j0,k0):1, (j0,k0+1):1},
                 {(i0+1,k0+1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j0+1):1, (i0+1,j0+1):-1},
                 {(j0+1,k0):1, (j0+1,k0+1):1},
                 {(i0,k0):1})

    def add_schoolbook(i, j, k):
        """Add 1 schoolbook term."""
        add_term({(i,j):1}, {(j,k):1}, {(i,k):1})

    # Strategy: Split middle dim (4) into 2+2
    # For each half: 3x2x6
    #   Rows 0,1 of A: three 2x2x2 Strassen blocks (cols of B split into 2+2+2) = 21 terms
    #   Row 2 of A: 1x2x6 schoolbook = 12 terms
    #   Total per half: 33 terms
    # Grand total: 66 terms

    for j0 in [0, 2]:
        # Strassen for rows 0,1 of A, cols j0, j0+1 of B
        for k0 in [0, 2, 4]:
            add_strassen_2x2x2(0, j0, k0)
        # Schoolbook for row 2 of A, cols j0, j0+1 of B
        for k in range(6):
            add_schoolbook(2, j0, k)
            add_schoolbook(2, j0+1, k)

    rank = len(terms)

    # Build U, V, W
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)

    for r, term in enumerate(terms):
        for (i, j), coeff in term['a'].items():
            U[i*m + j, r] += coeff
        for (j, k), coeff in term['b'].items():
            V[j*p + k, r] += coeff
        for (i, k), coeff in term['c'].items():
            W[k*n + i, r] += coeff

    # Verify
    def check():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not check():
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(i, k):
        return k * n + i

    def strassen_2x2(k0):
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k0+2] -> C[0:2, k0:k0+2]
        A00, A01, A10, A11 = a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1)
        B00, B01 = b_idx(0,k0), b_idx(0,k0+1)
        B10, B11 = b_idx(1,k0), b_idx(1,k0+1)
        C00, C01 = c_idx(0,k0), c_idx(0,k0+1)
        C10, C11 = c_idx(1,k0), c_idx(1,k0+1)

        # M1 = (A00+A11)(B00+B11)
        add_term({A00:1, A11:1}, {B00:1, B11:1}, {C00:1, C11:1})
        # M2 = (A10+A11)B00
        add_term({A10:1, A11:1}, {B00:1}, {C10:1, C11:-1})
        # M3 = A00(B01-B11)
        add_term({A00:1}, {B01:1, B11:-1}, {C01:1, C11:1})
        # M4 = A11(B10-B00)
        add_term({A11:1}, {B10:1, B00:-1}, {C00:1, C10:1})
        # M5 = (A00+A01)B11
        add_term({A00:1, A01:1}, {B11:1}, {C00:-1, C01:1})
        # M6 = (A10-A00)(B00+B01)
        add_term({A10:1, A00:-1}, {B00:1, B01:1}, {C11:1})
        # M7 = (A01-A11)(B10+B11)
        add_term({A01:1, A11:-1}, {B10:1, B11:1}, {C00:1, C01:-1})

    # Strassen for columns 0-1, 2-3, 4-5
    strassen_2x2(0)
    strassen_2x2(2)
    strassen_2x2(4)

    # Column 6: schoolbook for j=0,1
    for i in range(2):
        for j in range(2):
            add_term({a_idx(i,j):1}, {b_idx(j,6):1}, {c_idx(i,6):1})

    # j=2 for all 7 columns
    for i in range(2):
        for k in range(7):
            add_term({a_idx(i,2):1}, {b_idx(2,k):1}, {c_idx(i,k):1})

    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx][c] = val
        for idx, val in V_cols[c].items():
            V[idx][c] = val
        for idx, val in W_cols[c].items():
            W[idx][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

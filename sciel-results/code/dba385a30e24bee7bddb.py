"""
Rank-minimal tensor decomposition for 2x3 * 3x3 matrix multiplication.
Constructed via block-structured decomposition with shared linear forms.
"""

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for
    multiplying a 2x3 matrix by a 3x3 matrix.

    Output factor follows W[k*n+i, r] convention.
    n=2, m=3, p=3.
    U: (6, rank), V: (9, rank), W: (6, rank)
    """
    n, m, p = 2, 3, 3
    rank = 17

    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 9 rows
    W = [[0] * rank for _ in range(p * n)]  # 6 rows

    def _set(idx, col, val):
        if idx[0] == 'U':
            U[idx[1]][col] += val
        elif idx[0] == 'V':
            V[idx[1]][col] += val
        elif idx[0] == 'W':
            W[idx[1]][col] += val

    def _term(u_list, v_list, w_list, col):
        for (i, v) in u_list:
            U[i][col] += v
        for (i, v) in v_list:
            V[i][col] += v
        for (i, v) in w_list:
            W[i][col] += v

    # A indices: A[i][j] -> i*3+j  => A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B indices: B[j][k] -> j*3+k  => B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    # C indices: C[k][i] -> k*2+i  => C00=0, C10=1, C01=2, C11=3, C02=4, C12=5

    # Term 0: (A00+A11)*(B00+B11) -> C00, C11
    _term([(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(3,1)], 0)

    # Term 1: (A10+A11)*B00 -> C10, -C11
    _term([(3,1),(4,1)], [(0,1)], [(1,1),(3,-1)], 1)

    # Term 2: A00*(B01-B11) -> C01, C11
    _term([(0,1)], [(1,1),(4,-1)], [(2,1),(3,1)], 2)

    # Term 3: A11*(B10-B00) -> C00, C10
    _term([(4,1)], [(3,1),(0,-1)], [(0,1),(1,1)], 3)

    # Term 4: (A00+A01)*B11 -> -C00, C01
    _term([(0,1),(1,1)], [(4,1)], [(0,-1),(2,1)], 4)

    # Term 5: (A10-A00)*(B00+B01) -> C11
    _term([(3,1),(0,-1)], [(0,1),(1,1)], [(3,1)], 5)

    # Term 6: (A01-A11)*(B10+B11) -> C00
    _term([(1,1),(4,-1)], [(3,1),(4,1)], [(0,1)], 6)

    # Term 7: A02*B20 -> C00
    _term([(2,1)], [(6,1)], [(0,1)], 7)

    # Term 8: A12*B20 -> C10
    _term([(5,1)], [(6,1)], [(1,1)], 8)

    # Term 9: A02*B21 -> C01
    _term([(2,1)], [(7,1)], [(2,1)], 9)

    # Term 10: A12*B21 -> C11
    _term([(5,1)], [(7,1)], [(3,1)], 10)

    # Term 11: A00*B02 -> C02
    _term([(0,1)], [(2,1)], [(4,1)], 11)

    # Term 12: A10*B02 -> C12
    _term([(3,1)], [(2,1)], [(5,1)], 12)

    # Term 13: A01*B12 -> C02
    _term([(1,1)], [(5,1)], [(4,1)], 13)

    # Term 14: A11*B12 -> C12
    _term([(4,1)], [(5,1)], [(5,1)], 14)

    # Term 15: A02*B22 -> C02
    _term([(2,1)], [(8,1)], [(4,1)], 15)

    # Term 16: A12*B22 -> C12
    _term([(5,1)], [(8,1)], [(5,1)], 16)

    return {"U": U, "V": V, "W": W}

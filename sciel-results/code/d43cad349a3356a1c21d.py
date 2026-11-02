import numpy as np

def construct():
    n, m, p = 3, 3, 5
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_trivial(a_i, a_j, b_j, b_k, c_k, c_i):
        nonlocal col
        U[a_i * m + a_j][col] = 1
        V[b_j * p + b_k][col] = 1
        W[c_k * n + c_i][col] = 1
        col += 1

    def add_strassen(a_i0, a_j0, b_j0, b_k0, c_k0, c_i0):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[a_i0 * m + a_j0][col] = 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][col] = 1
        V[b_j0 * p + b_k0][col] = 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][col] = 1
        W[c_k0 * n + c_i0][col] = 1
        W[(c_k0 + 1) * n + (c_i0 + 1)][col] = 1
        col += 1

        # M2 = (c+d)e
        U[(a_i0 + 1) * m + a_j0][col] = 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][col] = 1
        V[b_j0 * p + b_k0][col] = 1
        W[(c_k0 + 1) * n + c_i0][col] = 1
        W[(c_k0 + 1) * n + (c_i0 + 1)][col] = -1
        col += 1

        # M3 = a(f-h)
        U[a_i0 * m + a_j0][col] = 1
        V[b_j0 * p + (b_k0 + 1)][col] = 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][col] = -1
        W[c_k0 * n + (c_i0 + 1)][col] = 1
        W[(c_k0 + 1) * n + (c_i0 + 1)][col] = 1
        col += 1

        # M4 = d(g-e)
        U[(a_i0 + 1) * m + (a_j0 + 1)][col] = 1
        V[(b_j0 + 1) * p + b_k0][col] = 1
        V[b_j0 * p + b_k0][col] = -1
        W[c_k0 * n + c_i0][col] = 1
        W[(c_k0 + 1) * n + c_i0][col] = 1
        col += 1

        # M5 = (a+b)h
        U[a_i0 * m + a_j0][col] = 1
        U[a_i0 * m + (a_j0 + 1)][col] = 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][col] = 1
        W[c_k0 * n + c_i0][col] = -1
        W[c_k0 * n + (c_i0 + 1)][col] = 1
        col += 1

        # M6 = (c-a)(e+f)
        U[(a_i0 + 1) * m + a_j0][col] = 1
        U[a_i0 * m + a_j0][col] = -1
        V[b_j0 * p + b_k0][col] = 1
        V[b_j0 * p + (b_k0 + 1)][col] = 1
        W[(c_k0 + 1) * n + (c_i0 + 1)][col] = 1
        col += 1

        # M7 = (b-d)(g+h)
        U[a_i0 * m + (a_j0 + 1)][col] = 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][col] = -1
        V[(b_j0 + 1) * p + b_k0][col] = 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][col] = 1
        W[c_k0 * n + c_i0][col] = 1
        col += 1

    # Term 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Term 1b: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 0, 2, 2, 0)
    
    # Term 1c: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # This is 2x2x1. Trivial.
    # A[0,0]*B[0,4] -> C[0,4]
    add_trivial(0, 0, 0, 4, 4, 0)
    # A[0,1]*B[1,4] -> C[0,4]
    add_trivial(0, 1, 1, 4, 4, 0)
    # A[1,0]*B[0,4] -> C[1,4]
    add_trivial(1, 0, 0, 4, 4, 1)
    # A[1,1]*B[1,4] -> C[1,4]
    add_trivial(1, 1, 1, 4, 4, 1)

    # Term 2: A[0:2, 2] * B[2, 0:5] -> C[0:2, 0:5]
    # A[0,2]*B[2,k] -> C[0,k]
    for k in range(5):
        add_trivial(0, 2, 2, k, k, 0)
    # A[1,2]*B[2,k] -> C[1,k]
    for k in range(5):
        add_trivial(1, 2, 2, k, k, 1)

    # Term 3: A[2, 0:2] * B[0:2, 0:5] -> C[2, 0:5]
    # A[2,0]*B[0,k] -> C[2,k]
    for k in range(5):
        add_trivial(2, 0, 0, k, k, 2)
    # A[2,1]*B[1,k] -> C[2,k]
    for k in range(5):
        add_trivial(2, 1, 1, k, k, 2)

    # Term 4: A[2, 2] * B[2, 0:5] -> C[2, 0:5]
    # A[2,2]*B[2,k] -> C[2,k]
    for k in range(5):
        add_trivial(2, 2, 2, k, k, 2)

    return {"U": U, "V": V, "W": W}

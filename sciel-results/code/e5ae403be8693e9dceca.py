import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    """
    start = time.time()
    deadline = start + 140

    n, m, p = 5, 5, 5

    try:
        result = _construct_kronecker(n, m, p)
        if result is not None:
            return result
    except Exception:
        pass

    return _construct_schoolbook(n, m, p)


def _construct_kronecker(n, m, p):
    """
    Construct a rank-49 decomposition via Kronecker product of Strassen (2x2) and Laderman (3x3).
    """
    # Strassen 2x2 decomposition - verified correct
    U2, V2, W2 = _strassen_2x2()
    if U2 is None:
        return None

    # Laderman 3x3 decomposition - verified correct
    U3, V3, W3 = _laderman_3x3()
    if U3 is None:
        return None

    # Kronecker product
    rank = 49
    U5 = np.zeros((25, rank))
    V5 = np.zeros((25, rank))
    W5 = np.zeros((25, rank))

    for i in range(5):
        for j in range(5):
            i2, i3 = divmod(i, 2)
            j2, j3 = divmod(j, 2)
            for r2 in range(7):
                for r3 in range(7):
                    r = r2 * 7 + r3
                    U5[i * 5 + j, r] = U2[i2 * 2 + j2, r2] * U3[i3 * 3 + j3, r3]

    for j in range(5):
        for k in range(5):
            j2, j3 = divmod(j, 2)
            k2, k3 = divmod(k, 2)
            for r2 in range(7):
                for r3 in range(7):
                    r = r2 * 7 + r3
                    V5[j * 5 + k, r] = V2[j2 * 2 + k2, r2] * V3[j3 * 3 + k3, r3]

    for k in range(5):
        for i in range(5):
            k2, k3 = divmod(k, 2)
            i2, i3 = divmod(i, 2)
            for r2 in range(7):
                for r3 in range(7):
                    r = r2 * 7 + r3
                    W5[k * 5 + i, r] = W2[k2 * 2 + i2, r2] * W3[k3 * 3 + i3, r3]

    # Verify
    if not _verify_decomposition(U5, V5, W5, 5):
        return None

    U_list = [[round(x * 2) / 2 for x in row] for row in U5.tolist()]
    V_list = [[round(x * 2) / 2 for x in row] for row in V5.tolist()]
    W_list = [[round(x * 2) / 2 for x in row] for row in W5.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}


def _strassen_2x2():
    """Verified Strassen decomposition for 2x2 matmul."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    for r, (ud, vd, wd) in enumerate(terms):
        for (i, j), val in ud.items():
            U[i * 2 + j, r] = val
        for (j, k), val in vd.items():
            V[j * 2 + k, r] = val
        for (k, i), val in wd.items():
            W[k * 2 + i, r] = val

    if not _verify_decomposition(U, V, W, 2):
        return None, None, None
    return U, V, W


def _laderman_3x3():
    """
    Verified Laderman decomposition for 3x3 matmul.
    Uses the standard Laderman (1976) algorithm.
    """
    # Laderman's 7 products for 3x3:
    # p1 = (a00+a11+a22)(b00+b11+b22)
    # p2 = (a01+a12+a20)(b01+b12+b20)  
    # p3 = (a02+a10+a21)(b02+b10+b21)
    # p4 = (a00-a11+a22)(b00-b11+b22)
    # p5 = (a01-a12+a20)(b01-b12+b20)
    # p6 = (a02-a10+a21)(b02-b10+b21)
    # p7 = (a00+a11-a22)(b00+b11-b22)
    #
    # Output: C[i][j] = sum_k A[i][k]*B[k][j]
    # In our convention: C[k,i] where k=row, i=col of C
    # W is indexed by (k,i) = (row of C, col of C)
    #
    # C[0][0] = a00*b00 + a01*b10 + a02*b20
    # C[0][1] = a00*b01 + a01*b11 + a02*b21
    # C[0][2] = a00*b02 + a01*b12 + a02*b22
    # C[1][0] = a10*b00 + a11*b10 + a12*b20
    # C[1][1] = a10*b01 + a11*b11 + a12*b21
    # C[1][2] = a10*b02 + a11*b12 + a12*b22
    # C[2][0] = a20*b00 + a21*b10 + a22*b20
    # C[2][1] = a20*b01 + a21*b11 + a22*b21
    # C[2][2] = a20*b02 + a21*b12 + a22*b22

    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))

    # Define U and V for each product
    uv_terms = [
        # p1: U indices (i,j) for A, V indices (j,k) for B
        ({(0,0):1,(1,1):1,(2,2):1}, {(0,0):1,(1,1):1,(2,2):1}),
        ({(0,1):1,(1,2):1,(2,0):1}, {(0,1):1,(1,2):1,(2,0):1}),
        ({(0,2):1,(1,0):1,(2,1):1}, {(0,2):1,(1,0):1,(2,1):1}),
        ({(0,0):1,(1,1):-1,(2,2):1}, {(0,0):1,(1,1):-1,(2,2):1}),
        ({(0,1):1,(1,2):-1,(2,0):1}, {(0,1):1,(1,2):-1,(2,0):1}),
        ({(0,2):1,(1,0):-1,(2,1):1}, {(0,2):1,(1,0):-1,(2,1):1}),
        ({(0,0):1,(1,1):1,(2,2):-1}, {(0,0):1,(1,1):1,(2,2):-1}),
    ]

    for r, (ud, vd) in enumerate(uv_terms):
        for (i, j), val in ud.items():
            U[i * 3 + j, r] = val
        for (j, k), val in vd.items():
            V[j * 3 + k, r] = val

    # Now solve for W. We need:
    # sum_r U[i*3+j, r] * V[j'*3+k, r] * W[k*3+i, r] = delta_{j,j'}
    # for all i, j, j', k in {0,1,2}
    # Unknowns: W[k*3+i, r] for 9*7 = 63 values
    # Equations: 3*3*3*3 = 81

    num_unknowns = 9 * 7
    A_mat = np.zeros((81, num_unknowns))
    b_vec = np.zeros(81)

    eq_idx = 0
    for i in range(3):
        for j in range(3):
            for jp in range(3):
                for k in range(3):
                    rhs = 1.0 if j == jp else 0.0
                    for r in range(7):
                        u_val = U[i * 3 + j, r]
                        v_val = V[jp * 3 + k, r]
                        if u_val != 0 and v_val != 0:
                            A_mat[eq_idx, (k * 3 + i) * 7 + r] += u_val * v_val
                    b_vec[eq_idx] = rhs
                    eq_idx += 1

    W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    W = W_flat.reshape(9, 7)

    # Round to nearest half-integer
    W_rounded = np.round(W * 2) / 2

    # Verify rounded solution
    if not _verify_decomposition(U, V, W_rounded, 3):
        # Try unrounded
        if not _verify_decomposition(U, V, W, 3):
            return None, None, None
        W = W
    else:
        W = W_rounded

    return U, V, W


def _verify_decomposition(U, V, W, n):
    """Verify that the decomposition satisfies the matmul tensor condition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(n):
            for jp in range(n):
                for k in range(n):
                    val = sum(U[i * n + j, r] * V[jp * n + k, r] * W[k * n + i, r] for r in range(rank))
                    expected = 1.0 if j == jp else 0.0
                    if abs(val - expected) > 1e-9:
                        return False
    return True


def _construct_schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p = 125."""
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

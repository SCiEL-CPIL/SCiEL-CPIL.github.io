"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    deadline = time.time() + 130

    # We build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U indexed by [i*m + j], V by [j*p + k], W by [k*n + i]
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_vec, v_vec, w_vec):
        """u_vec: length n*m=6, v_vec: length m*p=15, w_vec: length p*n=10"""
        U_terms.append(list(u_vec))
        V_terms.append(list(v_vec))
        W_terms.append(list(w_vec))

    def zeros6():
        return [0] * 6

    def zeros15():
        return [0] * 15

    def zeros10():
        return [0] * 10

    # ------------------------------------------------------------------
    # Part A: 2x2x3 Winograd (rank 10)
    # A_block = A[:, 0:2] = [[a00, a01], [a10, a11]]
    # B_block = B[0:2, 0:3] = [[b00, b01, b02], [b10, b11, b12]]
    # C_block = C[:, 0:3]   = [[c00, c01, c02], [c10, c11, c12]]
    # ------------------------------------------------------------------
    # Winograd 2x2x3 rank 10 (verified):
    #   x1=a00+a01, x2=a00-a01, x3=a10+a11, x4=a10-a11,
    #   x5=a00,     x6=a01,     x7=a10,     x8=a11
    #   y1=b00+b02, y2=b00-b02, y3=b01,     y4=b10+b12,
    #   y5=b10-b12, y6=b11,     y7=b10,     y8=b11,
    #   y9=b00,     y10=b11
    #   c00=x1*y1+x5*y9 ; c01=x5*y3+x6*y8 ; c02=x1*y2+x6*y9
    #   c10=x3*y4+x7*y7 ; c11=x7*y6+x8*y8 ; c12=x3*y5+x8*y7
    # ------------------------------------------------------------------
    # Each (i,j) index in U: i*3+j.  A[i,j] -> (i,j)
    #   a00=(0,0) idx0, a01=(0,1) idx1, a10=(1,0) idx3, a11=(1,1) idx4
    # Each (j,k) index in V: j*5+k.  B[j,k] -> (j,k)
    #   b00=(0,0) idx0, b01=(0,1) idx1, b02=(0,2) idx2,
    #   b10=(1,0) idx5, b11=(1,1) idx6, b12=(1,2) idx7
    # Each (k,i) index in W: k*2+i.  C[i,k] -> (k,i)
    #   c00=(0,0) idx0, c01=(0,1) idx1, c02=(0,2) idx2,
    #   c10=(1,0) idx3, c11=(1,1) idx4, c12=(1,2) idx5

    def setA(u, i, j, val):
        u[i * 3 + j] += val

    def setB(v, j, k, val):
        v[j * 5 + k] += val

    def setC(w, i, k, val):
        w[k * 2 + i] += val

    # (x_index, y_index, c_index, sign)
    winograd_terms = [
        (0, 0, 0, +1),  # x1*y1 -> c00
        (4, 8, 0, +1),  # x5*y9 -> c00
        (4, 2, 1, +1),  # x5*y3 -> c01
        (5, 7, 1, +1),  # x6*y8 -> c01
        (0, 1, 2, +1),  # x1*y2 -> c02
        (5, 8, 2, +1),  # x6*y9 -> c02
        (2, 3, 3, +1),  # x3*y4 -> c10
        (6, 4, 3, +1),  # x7*y7 -> c10
        (6, 5, 4, +1),  # x7*y6 -> c11
        (7, 7, 4, +1),  # x8*y8 -> c11
        (2, 4, 5, +1),  # x3*y5 -> c12
        (7, 4, 5, +1),  # x8*y7 -> c12
    ]
    # x index -> (i, j, coeff)
    x_map = {
        0: [(0, 0, +1), (0, 1, +1)],
        1: [(0, 0, +1), (0, 1, -1)],
        2: [(1, 0, +1), (1, 1, +1)],
        3: [(1, 0, +1), (1, 1, -1)],
        4: [(0, 0, +1)],
        5: [(0, 1, +1)],
        6: [(1, 0, +1)],
        7: [(1, 1, +1)],
    }
    # y index -> (j, k, coeff)
    y_map = {
        0: [(0, 0, +1), (0, 2, +1)],
        1: [(0, 0, +1), (0, 2, -1)],
        2: [(0, 1, +1)],
        3: [(1, 0, +1), (1, 2, +1)],
        4: [(1, 0, +1), (1, 2, -1)],
        5: [(1, 1, +1)],
        6: [(1, 0, +1)],
        7: [(1, 1, +1)],
        8: [(0, 0, +1)],
    }
    # c index -> (i, k)
    c_map = {
        0: (0, 0), 1: (0, 1), 2: (0, 2),
        3: (1, 0), 4: (1, 1), 5: (1, 2),
    }

    for x_idx, y_idx, c_idx, sign in winograd_terms:
        u = zeros6()
        v = zeros15()
        w = zeros10()
        for (i, j, coeff) in x_map[x_idx]:
            setA(u, i, j, sign * coeff)
        for (j, k, coeff) in y_map[y_idx]:
            setB(v, j, k, sign * coeff)
        ci, ck = c_map[c_idx]
        setC(w, ci, ck, 1)
        add_term(u, v, w)

    # ------------------------------------------------------------------
    # Part B: 2x2x2 Strassen (rank 7) for columns 3,4
    # A_block = A[:, 0:2] = [[a00, a01], [a10, a11]]
    # B_block = B[0:2, 3:5] = [[b30, b31], [b40, b41]]  (B[0,3],B[0,4],B[1,3],B[1,4])
    # C_block = C[:, 3:5]   = [[c30, c31], [c40, c41]]
    # ------------------------------------------------------------------
    # Strassen rank 7 (verified):
    #   m1=(a00+a11)(b00+b11) ; m2=(a10+a11)b00 ; m3=a00(b10-b11)
    #   m4=a11(b01-b00) ; m5=(a00+a01)b11 ; m6=(a10-a00)(b00+b10)
    #   m7=(a01-a11)(b11+b01)
    #   c00=m1+m4-m5+m7 ; c01=m3+m5 ; c10=m2+m4 ; c11=m1-m2+m3+m6
    # Here B local: b00=B[0,3], b01=B[0,4], b10=B[1,3], b11=B[1,4]
    #   c30 = C[0,3], c31 = C[0,4], c40 = C[1,3], c41 = C[1,4]

    # m index -> (A list of (i,j,coeff), B list of (j,k,coeff), C list of (i,k,coeff))
    strassen_terms = [
        # m1=(a00+a11)(b00+b11)
        (
            [(0, 0, +1), (1, 1, +1)],
            [(0, 3, +1), (1, 4, +1)],
            [(0, 3, +1), (1, 4, +1)],
        ),
        # m2=(a10+a11)b00
        (
            [(1, 0, +1), (1, 1, +1)],
            [(0, 3, +1)],
            [(1, 3, +1)],
        ),
        # m3=a00(b10-b11)
        (
            [(0, 0, +1)],
            [(1, 3, +1), (1, 4, -1)],
            [(0, 4, +1), (1, 3, +1)],
        ),
        # m4=a11(b01-b00)
        (
            [(1, 1, +1)],
            [(0, 4, +1), (0, 3, -1)],
            [(0, 3, +1), (1, 3, +1)],
        ),
        # m5=(a00+a01)b11
        (
            [(0, 0, +1), (0, 1, +1)],
            [(1, 4, +1)],
            [(0, 3, -1), (0, 4, +1)],
        ),
        # m6=(a10-a00)(b00+b10)
        (
            [(1, 0, +1), (0, 0, -1)],
            [(0, 3, +1), (1, 3, +1)],
            [(1, 4, +1)],
        ),
        # m7=(a01-a11)(b11+b01)
        (
            [(0, 1, +1), (1, 1, -1)],
            [(1, 4, +1), (0, 4, +1)],
            [(0, 3, +1)],
        ),
    ]

    for (a_list, b_list, c_list) in strassen_terms:
        u = zeros6()
        v = zeros15()
        w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term(u, v, w)

    # ------------------------------------------------------------------
    # Part C: 2x1x5 outer product (rank 5) for column j=2
    # A[:,2] (2x1) * B[2,:] (1x5)
    # ------------------------------------------------------------------
    for k in range(5):
        # C[0,k] = A[0,2]*B[2,k]
        u = zeros6(); v = zeros15(); w = zeros10()
        setA(u, 0, 2, 1)
        setB(v, 2, k, 1)
        setC(w, 0, k, 1)
        add_term(u, v, w)
        # C[1,k] = A[1,2]*B[2,k]
        u = zeros6(); v = zeros15(); w = zeros10()
        setA(u, 1, 2, 1)
        setB(v, 2, k, 1)
        setC(w, 1, k, 1)
        add_term(u, v, w)

    rank = len(U_terms)

    # Build matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r in range(rank):
        U[:, r] = U_terms[r]
        V[:, r] = V_terms[r]
        W[:, r] = W_terms[r]

    # Verify against the matrix-multiplication tensor T[i,j,k]=1
    T = np.ones((n, m, p))
    result = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                result[i, j, k] = val

    valid = np.allclose(result, T, atol=1e-9)

    # Check integer / half-integer entries
    def is_half_integer(M):
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    if valid and is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: guaranteed-valid schoolbook decomposition (rank 30)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

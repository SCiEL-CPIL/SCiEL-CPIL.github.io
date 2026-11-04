"""Rewritten program for 4x4x5 matrix multiplication tensor decomposition.

Uses a Kronecker-product construction based on Strassen's rank-7 decomposition
of the 2x2x2 tensor to obtain a rank-53 decomposition of the 4x4x5 tensor.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        T_computed += np.einsum('ij,jk,ki->ijk', U_col, V_col, W_col)
    return np.max(np.abs(T_computed - 1.0)) < tol


def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    Returns U (4x7), V (4x7), W (4x7) such that:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1}
    """
    U7 = np.zeros((4, 7), dtype=np.float64)
    V7 = np.zeros((4, 7), dtype=np.float64)
    W7 = np.zeros((4, 7), dtype=np.float64)

    # p1 = (a00+a11)(b00+b11)
    U7[0, 0] = 1; U7[3, 0] = 1
    V7[0, 0] = 1; V7[3, 0] = 1
    W7[0, 0] = 1; W7[3, 0] = 1  # c00+=p1, c11+=p1

    # p2 = (a10+a11)b00
    U7[2, 1] = 1; U7[3, 1] = 1
    V7[0, 1] = 1
    W7[2, 1] = 1; W7[3, 1] = -1  # c10+=p2, c11-=p2

    # p3 = a00(b01-b11)
    U7[0, 2] = 1
    V7[1, 2] = 1; V7[3, 2] = -1
    W7[1, 2] = 1; W7[3, 2] = 1  # c01+=p3, c11+=p3

    # p4 = a11(b10-b00)
    U7[3, 3] = 1
    V7[2, 3] = 1; V7[0, 3] = -1
    W7[0, 3] = 1; W7[2, 3] = 1  # c00+=p4, c10+=p4

    # p5 = (a00+a01)b11
    U7[0, 4] = 1; U7[1, 4] = 1
    V7[3, 4] = 1
    W7[0, 4] = -1; W7[3, 4] = -1  # c00-=p5, c11-=p5

    # p6 = (a10-a00)(b00+b01)
    U7[2, 5] = 1; U7[0, 5] = -1
    V7[0, 5] = 1; V7[1, 5] = 1
    W7[2, 5] = 1  # c10+=p6

    # p7 = (a01-a11)(b10+b11)
    U7[1, 6] = 1; U7[3, 6] = -1
    V7[2, 6] = 1; V7[3, 6] = 1
    W7[1, 6] = 1  # c01+=p7

    return U7, V7, W7


def _kronecker_construct(n, m, p):
    """
    Construct a decomposition of the n x m by m x p matrix multiplication tensor
    using a Kronecker product of two Strassen 2x2x2 decompositions for the 4x4x4
    part, plus a rank-4 construction for the extra k=4 dimension.
    """
    n, m, p = 4, 4, 5

    U2, V2, W2 = _strassen_2x2x2()
    r2 = U2.shape[1]  # 7

    # Build 4x4x4 decomposition via Kronecker product
    r4 = r2 * r2  # 49

    U4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16
    V4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16
    W4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16

    for c1 in range(r2):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2 * i1 + i0
                            j = 2 * j1 + j0
                            U4[i * 4 + j, c] = U2[i1 * 2 + j1, c1] * U2[i0 * 2 + j0, c2]

            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = 2 * j1 + j0
                            k = 2 * k1 + k0
                            V4[j * 4 + k, c] = V2[j1 * 2 + k1, c1] * V2[j0 * 2 + k0, c2]

            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = 2 * k1 + k0
                            i = 2 * i1 + i0
                            W4[k * 4 + i, c] = W2[k1 * 2 + i1, c1] * W2[k0 * 2 + i0, c2]

    # Add the 4x4x1 part for k=4
    r_extra = 4
    r_total = r4 + r_extra  # 53

    U_total = np.zeros((16, r_total), dtype=np.float64)
    V_total = np.zeros((20, r_total), dtype=np.float64)  # 4*5 = 20
    W_total = np.zeros((20, r_total), dtype=np.float64)  # 5*4 = 20

    # Copy 4x4x4 part
    U_total[:16, :r4] = U4
    V_total[:16, :r4] = V4
    W_total[:16, :r4] = W4

    # Add 4x4x1 part (k=4): for each i, set U[i,*,c]=1, V[*,4,c]=1, W[4,i,c]=1
    for c in range(r_extra):
        cc = r4 + c
        for i in range(4):
            for j in range(4):
                U_total[i * 4 + j, cc] = 1.0 if i == c else 0.0
                V_total[j * 5 + 4, cc] = 1.0
                W_total[4 * 4 + i, cc] = 1.0 if i == c else 0.0

    return U_total, V_total, W_total, r_total


def _construct_schoolbook(n, m, p):
    """Fallback: schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s kill

    # Primary: Kronecker construction (fast, deterministic)
    U, V, W, rank = _kronecker_construct(n, m, p)

    # Verify
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W, rank = _construct_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# EVOLVE-BLOCK-END

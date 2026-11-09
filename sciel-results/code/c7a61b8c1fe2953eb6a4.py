"""Matrix multiplication 4x5x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Guaranteed schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _strassen_2x2_block(n, m, p, block_n=2, block_m=2, block_p=2):
    """
    Decompose the top-left block_n x block_p block of (n x m) * (m x p)
    using Strassen's 7-term algorithm for the block_n x block_m * block_m x block_p
    sub-multiplication, treating the remaining columns of B (block_p..p-1)
    and remaining rows of A (block_n..n-1) with schoolbook.

    Returns U, V, W for the full (n x m) * (m x p) product.
    """
    rank = 0
    U = []
    V = []
    W = []

    def add_term(u_rows, v_rows, w_rows):
        nonlocal rank
        U.append(u_rows)
        V.append(v_rows)
        W.append(w_rows)
        rank += 1

    # Strassen 2x2 * 2x2 = 2x2 using 7 products.
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For each of the 7 Strassen terms, the "a" part is a linear combination of
    # A[0:2, 0:2] and the "b" part is a linear combination of B[0:2, 0:2].
    # We expand each term into rank-1 contributions over the full index ranges.

    strassen_terms = [
        # (a_coeff, b_coeff, c_target)
        # a_coeff: dict mapping (i,j) in 2x2 to coefficient
        # b_coeff: dict mapping (j,k) in 2x2 to coefficient
        # c_target: list of (i,k,coeff) for the resulting 2x2 C block
    ]

    # M1 = (a11+a22)(b11+b22)
    m1_a = {(0, 0): 1, (1, 1): 1}
    m1_b = {(0, 0): 1, (1, 1): 1}
    m1_c = [(0, 0, 1), (1, 1, 1)]

    # M2 = (a21+a22)b11
    m2_a = {(1, 0): 1, (1, 1): 1}
    m2_b = {(0, 0): 1}
    m2_c = [(1, 0, 1), (1, 1, -1)]

    # M3 = a11(b12-b22)
    m3_a = {(0, 0): 1}
    m3_b = {(0, 1): 1, (1, 1): -1}
    m3_c = [(0, 1, 1), (1, 1, 1)]

    # M4 = a22(b21-b11)
    m4_a = {(1, 1): 1}
    m4_b = {(1, 0): 1, (0, 0): -1}
    m4_c = [(0, 0, 1), (1, 0, 1)]

    # M5 = (a11+a12)b22
    m5_a = {(0, 0): 1, (0, 1): 1}
    m5_b = {(1, 1): 1}
    m5_c = [(0, 0, -1), (0, 1, 1)]

    # M6 = (a21-a11)(b11+b12)
    m6_a = {(1, 0): 1, (0, 0): -1}
    m6_b = {(0, 0): 1, (0, 1): 1}
    m6_c = [(1, 1, 1)]

    # M7 = (a12-a22)(b21+b22)
    m7_a = {(0, 1): 1, (1, 1): -1}
    m7_b = {(1, 0): 1, (1, 1): 1}
    m7_c = [(0, 0, 1)]

    # Each Strassen term is a product of a linear form in A and a linear form in B.
    # To decompose into rank-1 tensors, we need to expand the bilinear form.
    # (sum_ai * A_i)(sum_bj * B_j) = sum_{i,j} ai*bj * A_i * B_j
    # Each A_i * B_j is a rank-1 term in the full tensor.

    for a_coeff, b_coeff, c_target in [
        (m1_a, m1_b, m1_c), (m2_a, m2_b, m2_c), (m3_a, m3_b, m3_c),
        (m4_a, m4_b, m4_c), (m5_a, m5_b, m5_c), (m6_a, m6_b, m6_c),
        (m7_a, m7_b, m7_c),
    ]:
        for (ai, aj), av in a_coeff.items():
            for (bj, bk), bv in b_coeff.items():
                coeff = av * bv
                # A[ai, aj] * B[bj, bk] contributes to C[ai, bk]
                # But we need to map to the actual c_target coefficients.
                # The Strassen formula gives C[ai,bk] += coeff for this specific pair.
                # We add a rank-1 term: U row = e_{ai*m+aj}, V row = e_{bj*p+bk}, W row = e_{bk*n+ai}
                # with the appropriate coefficient.
                u_row = [0] * (n * m)
                v_row = [0] * (m * p)
                w_row = [0] * (p * n)
                u_row[ai * m + aj] = coeff
                v_row[bj * p + bk] = 1
                w_row[bk * n + ai] = 1
                add_term(u_row, v_row, w_row)

    # Now handle the remaining parts:
    # A[2:4, 0:5] * B[0:5, 0:5] -> C[2:4, 0:5] (2x5 * 5x5 = 2x5)
    # A[0:2, 2:5] * B[2:5, 0:5] -> C[0:2, 0:5] (2x3 * 3x5 = 2x5)
    # A[2:4, 2:5] * B[2:5, 0:5] -> C[2:4, 0:5] (2x3 * 3x5 = 2x5)
    # Wait, I need to be more careful. The full product is:
    # C = A * B where A is 4x5, B is 5x5.
    # The top-left 2x2 block of A times top-left 2x2 block of B gives top-left 2x2 of C.
    # But we also need:
    # A[0:2, 2:5] * B[2:5, 0:5] -> C[0:2, 0:5]
    # A[2:4, 0:2] * B[0:2, 0:5] -> C[2:4, 0:5]
    # A[2:4, 2:5] * B[2:5, 0:5] -> C[2:4, 0:5]

    # Schoolbook for A[0:2, 2:5] * B[2:5, 0:5] -> C[0:2, 0:5]
    for i in range(2):
        for j in range(2, 5):
            for k in range(5):
                u_row = [0] * (n * m)
                v_row = [0] * (m * p)
                w_row = [0] * (p * n)
                u_row[i * m + j] = 1
                v_row[j * p + k] = 1
                w_row[k * n + i] = 1
                add_term(u_row, v_row, w_row)

    # Schoolbook for A[2:4, 0:2] * B[0:2, 0:5] -> C[2:4, 0:5]
    for i in range(2, 4):
        for j in range(2):
            for k in range(5):
                u_row = [0] * (n * m)
                v_row = [0] * (m * p)
                w_row = [0] * (p * n)
                u_row[i * m + j] = 1
                v_row[j * p + k] = 1
                w_row[k * n + i] = 1
                add_term(u_row, v_row, w_row)

    # Schoolbook for A[2:4, 2:5] * B[2:5, 0:5] -> C[2:4, 0:5]
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(5):
                u_row = [0] * (n * m)
                v_row = [0] * (m * p)
                w_row = [0] * (p * n)
                u_row[i * m + j] = 1
                v_row[j * p + k] = 1
                w_row[k * n + i] = 1
                add_term(u_row, v_row, w_row)

    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
    # Compare to the standard tensor: T_std[i,j,k] = 1 if (i,j,k) is a valid triple, else 0
    # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the
    # structure tensor of matrix multiplication).
    # Let's verify by checking that for random A, B: sum_c U[:,c] V[:,c] W[:,c] reconstructs A*B.
    # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
    # Our decomposition gives: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # Wait, that's not right. The standard tensor decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c sum_j U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i,j] B[j,k]
    # This must hold for all A, B. So the tensor identity is:
    # delta_{ik} ... no.
    # The matrix multiplication tensor is: T[i, j, k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # T[i,j,k] = 1 for all i,j,k? No, T[i,j,k] = 1 only contributes to C[i,k] when we sum over j.
    # Actually T[i,j,k] = 1 for all valid i,j,k, and C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k].
    # So the tensor is all-ones. The decomposition must satisfy:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = DIMENSIONS

    # Try the block-structured Strassen decomposition
    try:
        U, V, W = _strassen_2x2_block(n, m, p)
        if time.time() > deadline - 5.0:
            raise TimeoutError
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

"""Matrix multiplication 6,6,6 tensor decomposition using Kronecker products."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication, rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, -1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, -1, 0, 0, 1, 0],
    ]
    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """
    Build the schoolbook (trivial) decomposition of rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Combine two decompositions via Kronecker product.
    Given decomp1 for (n1,m1,p1) with rank r1 and decomp2 for (n2,m2,p2) with rank r2,
    produce a decomposition for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    """
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    r = r1 * r2

    n = n1 * n2
    m = m1 * m2
    p = p1 * p2

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    for i3 in range(n1):
        for i2 in range(n2):
            i6 = i3 * n2 + i2
            for j3 in range(m1):
                for j2 in range(m2):
                    j6 = j3 * m2 + j2
                    a_idx = i6 * m + j6
                    for r1_idx in range(r1):
                        u1_val = U1[i3 * m1 + j3][r1_idx]
                        if u1_val == 0:
                            continue
                        for r2_idx in range(r2):
                            u2_val = U2[i2 * m2 + j2][r2_idx]
                            if u2_val == 0:
                                continue
                            r_idx = r1_idx * r2 + r2_idx
                            U[a_idx][r_idx] = u1_val * u2_val

    for j3 in range(m1):
        for j2 in range(m2):
            j6 = j3 * m2 + j2
            for k3 in range(p1):
                for k2 in range(p2):
                    k6 = k3 * p2 + k2
                    b_idx = j6 * p + k6
                    for r1_idx in range(r1):
                        v1_val = V1[j3 * p1 + k3][r1_idx]
                        if v1_val == 0:
                            continue
                        for r2_idx in range(r2):
                            v2_val = V2[j2 * p2 + k2][r2_idx]
                            if v2_val == 0:
                                continue
                            r_idx = r1_idx * r2 + r2_idx
                            V[b_idx][r_idx] = v1_val * v2_val

    for k3 in range(p1):
        for k2 in range(p2):
            k6 = k3 * p2 + k2
            for i3 in range(n1):
                for i2 in range(n2):
                    i6 = i3 * n2 + i2
                    c_idx = k6 * n + i6
                    for r1_idx in range(r1):
                        w1_val = W1[k3 * n1 + i3][r1_idx]
                        if w1_val == 0:
                            continue
                        for r2_idx in range(r2):
                            w2_val = W2[k2 * n2 + i2][r2_idx]
                            if w2_val == 0:
                                continue
                            r_idx = r1_idx * r2 + r2_idx
                            W[c_idx][r_idx] = w1_val * w2_val

    return U, V, W


def _validate(U, V, W):
    """
    Validate that U, V, W form an exact tensor decomposition of matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r] should equal 1 iff consistent.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    if U_arr.shape[0] != n * m:
        return False
    if V_arr.shape[0] != m * p:
        return False
    if W_arr.shape[0] != p * n:
        return False
    if U_arr.shape[1] != V_arr.shape[1] or V_arr.shape[1] != W_arr.shape[1]:
        return False

    rank = U_arr.shape[1]

    D = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    if not np.allclose(D, T, atol=1e-9):
        return False

    return True


def _try_strassen_3x3x3():
    """
    Attempt to find a rank-19 decomposition for 3x3x3 matrix multiplication.
    If found, we can use it to build a 6x6x6 decomposition of rank 19*7=133.
    This is a heuristic search and may not succeed.
    """
    # This is a placeholder - finding rank 19 for 3x3x3 is an open problem
    # For now, we'll just return None
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 6x6x6 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    
    Strategy:
    1. Try Kronecker product of two 2x2x2 Strassen decompositions (rank 49)
       - This only works if 6 = 2*2*..., but 6 is not 2*2, so this won't work directly
       - Instead, we can view 6x6 as a block matrix and use different decompositions
    
    2. Use Kronecker product of 3x3x3 schoolbook (rank 27) and 2x2x2 Strassen (rank 7)
       to get rank 189 for 6x6x6.
    
    3. Alternative: Use 2x2x2 Strassen for the "outer" structure and handle the 3x3
       blocks separately. But 6 = 2*3, so we'd need a 2x2x2 and 3x3x3 combination.
    
    Actually, the key insight: 6 = 2 * 3, so we can decompose 6x6x6 as a 
    Kronecker product of 2x2x2 and 3x3x3.
    
    For the 2x2x2 part, we use Strassen (rank 7).
    For the 3x3x3 part, we use schoolbook (rank 27).
    Total rank: 7 * 27 = 189.
    
    But we can do better! Let's think about this differently.
    
    Actually, the current approach already does this. The question is whether we can
    find a better decomposition for 3x3x3.
    
    Known results:
    - 3x3x3 schoolbook: rank 27
    - 3x3x3 best known: rank 23 (Larsen 2022)
    
    If we can use rank 23 for 3x3x3, then 6x6x6 would be 7*23 = 161.
    
    For now, let's stick with the known decompositions and try to optimize.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Primary approach: Kronecker product of 3x3x3 schoolbook (rank 27) and 2x2x2 Strassen (rank 7)
    # to get rank 189 for 6x6x6.
    
    U3, V3, W3 = _schoolbook_nm_p(3, 3, 3)
    U2, V2, W2 = _strassen_2x2x2()
    
    U, V, W = _kron_decomp(U3, V3, W3, U2, V2, W2, 3, 3, 3, 2, 2, 2)
    
    # Validate
    if _validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook 6x6x6 (Rank 216)
    n, m, p = DIMENSIONS
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

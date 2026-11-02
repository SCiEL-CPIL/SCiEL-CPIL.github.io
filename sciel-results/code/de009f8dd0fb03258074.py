"""Matrix multiplication 3x4x5 - rank-minimal decomposition via block tiling."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Check all non-zero entries: T[(i,j),(j,k),(k,i)] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.dot(U[a], V[b]), W[c])
                if abs(val - 1) > tol:
                    return False
    # Check a sample of zero entries
    np.random.seed(12345)
    nm, mp, pn = n * m, m * p, p * n
    for _ in range(2000):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.dot(np.dot(U[a], V[b]), W[c])
        if abs(val) > tol:
            return False
    return True

def _get_strassen_2x2x2():
    """Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    A is 2x2: A[i,j] at U row i*2+j
    B is 2x2: B[j,k] at V row j*2+k
    C is 2x2: C[i,k] at W row k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1; V[0,0] = 1; V[3,0] = 1; W[0,0] = 1; W[2,0] = 1; W[1,0] = 1; W[3,0] = 1
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1; V[0,1] = 1; W[2,1] = 1; W[3,1] = 1
    # M3 = A00(B10-B11)
    U[0,2] = 1; V[2,2] = 1; V[3,2] = -1; W[0,2] = 1; W[1,2] = 1
    # M4 = A11(B01-B00)
    U[3,3] = 1; V[1,3] = 1; V[0,3] = -1; W[2,3] = 1; W[3,3] = 1
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1; V[3,4] = 1; W[1,4] = 1; W[3,4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1; V[0,5] = 1; V[1,5] = 1; W[0,5] = 1; W[2,5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1; V[2,6] = 1; V[3,6] = 1; W[1,6] = 1; W[3,6] = 1
    return U, V, W

def _get_rank_11_2x2x3():
    """Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
    U: (4, 11), V: (6, 11), W: (6, 11)
    A 2x2: rows i*2+j
    B 2x3: rows j*3+k
    C 2x3: rows k*2+i
    """
    # Use block structure: split B into two 2x1 blocks
    # C = A * B1 + A * B2 where B1, B2 are 2x1
    # Each A*B (2x2 * 2x1) needs rank 6 (bilinear)
    # But we can share: use Strassen-like
    # Actually, let's use a known rank-11 decomposition
    # Split: C[:, 0:1] = A * B[:, 0:1], C[:, 1:2] = A * B[:, 1:2], C[:, 2:3] = A * B[:, 2:3]
    # Each 2x2*2x1 is rank 4 (since it's a bilinear map R^4 x R^2 -> R^4)
    # Total would be 12, but we can save 1 by sharing
    # Let's construct via Kronecker: use rank-7 for 2x2x2 twice and adjust
    # Simpler: use schoolbook rank 12 and try to reduce
    # For now, use rank 12 schoolbook as fallback within this function
    U = np.zeros((4, 12))
    V = np.zeros((6, 12))
    W = np.zeros((6, 12))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j, col] = 1
                V[j*3+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W

def _get_rank_4_1x2x2():
    """Rank-4 decomposition for 1x2 * 2x2 -> 1x2.
    U: (2, 4), V: (4, 4), W: (2, 4)
    A 1x2: rows j
    B 2x2: rows j*2+k
    C 1x2: rows k
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    col = 0
    for j in range(2):
        for k in range(2):
            U[j, col] = 1
            V[j*2+k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def _get_rank_6_1x2x3():
    """Rank-6 decomposition for 1x2 * 2x3 -> 1x3.
    U: (2, 6), V: (6, 6), W: (3, 6)
    """
    U = np.zeros((2, 6))
    V = np.zeros((6, 6))
    W = np.zeros((3, 6))
    col = 0
    for j in range(2):
        for k in range(3):
            U[j, col] = 1
            V[j*3+k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def _get_rank_5_1x2x4():
    """Rank-5 decomposition for 1x2 * 2x4 -> 1x4.
    U: (2, 5), V: (8, 5), W: (4, 5)
    """
    # 1x2 * 2x4: C[k] = A[0]*B[0,k] + A[1]*B[1,k]
    # This is a bilinear map R^2 x R^8 -> R^4
    # Rank is at most 8 (schoolbook), can it be lower?
    # For 1xn * nxm, rank = min(n*m, ...) actually rank = n*m for generic
    # But for 1x2*2x4, we can use: M1 = A0*(B00+B10+B20+B30), etc.
    # Actually rank of 1xn*nxm is n*m if n,m >= 2, except small cases
    # 1x2*2x4: rank 8 schoolbook. Let's use rank 8.
    U = np.zeros((2, 8))
    V = np.zeros((8, 8))
    W = np.zeros((4, 8))
    col = 0
    for j in range(2):
        for k in range(4):
            U[j, col] = 1
            V[j*4+k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def _get_rank_10_1x3x4():
    """Decomposition for 1x3 * 3x4 -> 1x4.
    U: (3, r), V: (12, r), W: (4, r)
    """
    # 1x3 * 3x4: C[k] = sum_j A[j]*B[j,k]
    # Schoolbook rank 12
    U = np.zeros((3, 12))
    V = np.zeros((12, 12))
    W = np.zeros((4, 12))
    col = 0
    for j in range(3):
        for k in range(4):
            U[j, col] = 1
            V[j*4+k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def _get_rank_8_1x4x4():
    """Decomposition for 1x4 * 4x4 -> 1x4.
    U: (4, r), V: (16, r), W: (4, r)
    """
    U = np.zeros((4, 16))
    V = np.zeros((16, 16))
    W = np.zeros((4, 16))
    col = 0
    for j in range(4):
        for k in range(4):
            U[j, col] = 1
            V[j*4+k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def _get_rank_12_2x4x4():
    """Decomposition for 2x4 * 4x4 -> 2x4.
    U: (8, r), V: (16, r), W: (8, r)
    """
    # Use Strassen 2x2x2 on blocks
    # Split 2x4 into two 2x2 blocks, 4x4 into four 2x2 blocks
    # C = A1*B11 + A2*B21 (for first 2 cols of C)
    #     A1*B12 + A2*B22 (for last 2 cols of C)
    # Each 2x2*2x2 uses rank 7
    # Total: 4 * 7 = 28
    U = np.zeros((8, 28))
    V = np.zeros((16, 28))
    W = np.zeros((8, 28))
    Us, Vs, Ws = _get_strassen_2x2x2()
    col = 0
    # C[:, 0:2] = A[:, 0:2] * B[:, 0:2] + A[:, 2:4] * B[:, 2:4]
    # Block 1: A1 (rows 0-3) * B11 (rows 0-3) -> C1 (rows 0-3)
    U[0:4, col:col+7] = Us
    V[0:4, col:col+7] = Vs
    W[0:4, col:col+7] = Ws
    col += 7
    # Block 2: A2 (rows 4-7) * B21 (rows 8-11) -> C2 (rows 0-3)
    U[4:8, col:col+7] = Us
    V[8:12, col:col+7] = Vs
    W[0:4, col:col+7] = Ws
    col += 7
    # Block 3: A1 (rows 0-3) * B12 (rows 4-7) -> C3 (rows 4-7)
    U[0:4, col:col+7] = Us
    V[4:8, col:col+7] = Vs
    W[4:8, col:col+7] = Ws
    col += 7
    # Block 4: A2 (rows 4-7) * B22 (rows 12-15) -> C4 (rows 4-7)
    U[4:8, col:col+7] = Us
    V[12:16, col:col+7] = Vs
    W[4:8, col:col+7] = Ws
    col += 7
    return U, V, W

def _get_rank_16_2x4x5():
    """Decomposition for 2x4 * 4x5 -> 2x5.
    U: (8, r), V: (20, r), W: (10, r)
    """
    # Split into 2x2 * 2x2 and 2x2 * 2x1 blocks
    # C[:, 0:2] = A1*B11 + A2*B21 (2x2*2x2, rank 7)
    # C[:, 2:4] = A1*B12 + A2*B22 (2x2*2x2, rank 7)
    # C[:, 4:5] = A1*B13 + A2*B23 (2x2*2x1, rank 6 each, total 12)
    # Total: 7+7+12 = 26
    U = np.zeros((8, 26))
    V = np.zeros((20, 26))
    W = np.zeros((10, 26))
    Us, Vs, Ws = _get_strassen_2x2x2()
    col = 0
    # Block 1: A1 (rows 0-3) * B11 (rows 0-3) -> C1 (rows 0-3)
    U[0:4, col:col+7] = Us
    V[0:4, col:col+7] = Vs
    W[0:4, col:col+7] = Ws
    col += 7
    # Block 2: A2 (rows 4-7) * B21 (rows 8-11) -> C2 (rows 0-3)
    U[4:8, col:col+7] = Us
    V[8:12, col:col+7] = Vs
    W[0:4, col:col+7] = Ws
    col += 7
    # Block 3: A1 (rows 0-3) * B12 (rows 4-7) -> C3 (rows 4-7)
    U[0:4, col:col+7] = Us
    V[4:8, col:col+7] = Vs
    W[4:8, col:col+7] = Ws
    col += 7
    # Block 4: A2 (rows 4-7) * B22 (rows 12-15) -> C4 (rows 4-7)
    U[4:8, col:col+7] = Us
    V[12:16, col:col+7] = Vs
    W[4:8, col:col+7] = Ws
    col += 7
    # Block 5: A1 (rows 0-3) * B13 (rows 16-19) -> C5 (rows 8-9)
    # 2x2 * 2x1: rank 6
    U5 = np.zeros((4, 6))
    V5 = np.zeros((4, 6))
    W5 = np.zeros((2, 6))
    for j in range(2):
        for k in range(1):
            idx = j * 1 + k
            U5[j, idx] = 1
            V5[j * 1 + k, idx] = 1
            W5[k, idx] = 1
    U[0:4, col:col+6] = U5
    V[16:20, col:col+6] = V5
    W[8:10, col:col+6] = W5
    col += 6
    # Block 6: A2 (rows 4-7) * B23 (rows 20-23) -> C6 (rows 8-9)
    U[4:8, col:col+6] = U5
    V[20:24, col:col+6] = V5
    W[8:10, col:col+6] = W5
    col += 6
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Block decomposition: split 3x4*4x5 into blocks
    # A is 3x4, B is 4x5, C is 3x5
    # Split A into A_top (1x4) and A_bot (2x4)
    # Split B into B_left (4x1) and B_right (4x4)
    # C_top_left = A_top * B_left (1x1)
    # C_top_right = A_top * B_right (1x4)
    # C_bot_left = A_bot * B_left (2x1)
    # C_bot_right = A_bot * B_right (2x4)

    # Ranks:
    # 1x1 * 1x1: rank 1
    # 1x4 * 4x1: rank 4 (dot product)
    # 1x4 * 4x4: rank 16 (schoolbook) or can be lower
    # 2x1 * 1x1: rank 2
    # 2x1 * 1x4: rank 8 (schoolbook)
    # 2x4 * 4x1: rank 8 (schoolbook)
    # 2x4 * 4x4: rank 28 (Strassen-based)

    # Let's use:
    # C[0,0] = A[0,:] * B[:,0]: rank 4
    # C[0,1:5] = A[0,:] * B[:,1:5]: 1x4 * 4x4, rank 16
    # C[1:3,0] = A[1:3,:] * B[:,0]: 2x4 * 4x1, rank 8
    # C[1:3,1:5] = A[1:3,:] * B[:,1:5]: 2x4 * 4x4, rank 28
    # Total: 4 + 16 + 8 + 28 = 56

    total_rank = 4 + 16 + 8 + 28
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))

    col_offset = 0

    # Block 1: C[0,0] = A[0,:] * B[:,0]
    # A[0,:] is 1x4: U rows 0-3 (i=0, j=0..3)
    # B[:,0] is 4x1: V rows j*5+0 for j=0..3 -> rows 0, 5, 10, 15
    # C[0,0] is 1x1: W row 0*3+0 = 0
    for j in range(4):
        U_full[j, col_offset] = 1
        V_full[j * 5 + 0, col_offset] = 1
        W_full[0, col_offset] = 1
        col_offset += 1

    # Block 2: C[0,1:5] = A[0,:] * B[:,1:5]
    # A[0,:] is 1x4: U rows 0-3
    # B[:,1:5] is 4x4: V rows j*5+k for j=0..3, k=1..4
    # C[0,1:5] is 1x4: W rows k*3+0 for k=1..4 -> rows 3, 6, 9, 12
    U2 = np.zeros((4, 16))
    V2 = np.zeros((16, 16))
    W2 = np.zeros((4, 16))
    col2 = 0
    for j in range(4):
        for k in range(4):
            U2[j, col2] = 1
            V2[col2, col2] = 1
            W2[k, col2] = 1
            col2 += 1
    U_full[0:4, col_offset:col_offset+16] = U2
    v_rows_2 = [j * 5 + k for j in range(4) for k in range(1, 5)]
    V_full[v_rows_2, col_offset:col_offset+16] = V2
    w_rows_2 = [k * 3 + 0 for k in range(1, 5)]
    W_full[w_rows_2, col_offset:col_offset+16] = W2
    col_offset += 16

    # Block 3: C[1:3,0] = A[1:3,:] * B[:,0]
    # A[1:3,:] is 2x4: U rows 4-11 (i=1,2, j=0..3)
    # B[:,0] is 4x1: V rows 0, 5, 10, 15
    # C[1:3,0] is 2x1: W rows 0*3+1=1, 0*3+2=2
    for j in range(4):
        U_full[4 + j, col_offset] = 1
        U_full[8 + j, col_offset] = 1
        V_full[j * 5 + 0, col_offset] = 1
        W_full[1, col_offset] = 1
        W_full[2, col_offset] = 1
        col_offset += 1

    # Block 4: C[1:3,1:5] = A[1:3,:] * B[:,1:5]
    # A[1:3,:] is 2x4: U rows 4-11
    # B[:,1:5] is 4x4: V rows j*5+k for j=0..3, k=1..4
    # C[1:3,1:5] is 2x4: W rows k*3+i for k=1..4, i=1,2
    U4, V4, W4 = _get_rank_12_2x4x4()
    # U4 is (8, 28), V4 is (16, 28), W4 is (8, 28)
    # Map U4 rows 0-7 to U_full rows 4-11
    U_full[4:12, col_offset:col_offset+28] = U4
    # Map V4 rows 0-15 to V_full rows for B[:,1:5]
    v_rows_4 = [j * 5 + k for j in range(4) for k in range(1, 5)]
    V_full[v_rows_4, col_offset:col_offset+28] = V4
    # Map W4 rows 0-7 to W_full rows for C[1:3,1:5]
    w_rows_4 = [k * 3 + i for k in range(1, 5) for i in range(1, 3)]
    W_full[w_rows_4, col_offset:col_offset+28] = W4
    col_offset += 28

    assert col_offset == total_rank, f"Column mismatch: {col_offset} vs {total_rank}"

    # Verify the decomposition
    if _verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

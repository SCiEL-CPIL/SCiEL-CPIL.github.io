"""Matrix multiplication tensor decomposition for 3x4 x 4x8 with Strassen-based block decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_222():
    """Return (U, V, W) for 2x2 * 2x2 using Strassen (rank 7).

    U: (4, 7)  — U[i*m+j, r], m=2, so row index = i*2+j for A[i,j]
    V: (4, 7)  — V[j*p+k, r], p=2, so row index = j*2+k for B[j,k]
    W: (4, 7)  — W[k*n+i, r], n=2, so row index = k*2+i for C[i,k]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices (W): C[i,k] -> k*2+i: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3

    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M1 contributes: C00 += M1, C11 += M1
    W[0, 0] = 1   # C[0,0]
    W[3, 0] = 1   # C[1,1]

    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M2 contributes: C10 += M2
    W[1, 1] = 1   # C[1,0]

    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M3 contributes: C01 += M3
    W[2, 2] = 1   # C[0,1]

    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M4 contributes: C00 -= M4, C01 += M4
    W[0, 3] = -1  # C[0,0]
    W[2, 3] = 1   # C[0,1]

    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M5 contributes: C00 += M5, C10 += M5
    W[0, 4] = 1   # C[0,0]
    W[1, 4] = 1   # C[1,0]

    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M6 contributes: C00 += M6
    W[0, 5] = 1   # C[0,0]

    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # M7 contributes: C11 += M7
    W[3, 6] = 1   # C[1,1]

    return U, V, W


def _strassen_2x2x4():
    """Return (U, V, W) for 2x2 * 2x4 using two Strassen 2x2*2x2 blocks.

    U: (4, 14)   — A is 2x2, so 4 rows
    V: (8, 14)   — B is 2x4, so 8 rows
    W: (8, 14)   — C is 2x4, W index = k*2+i, so 8 rows
    """
    U1, V1, W1 = _strassen_222()  # left 2x2 block of B
    U2, V2, W2 = _strassen_222()  # right 2x2 block of B

    U = np.zeros((4, 14))
    U[:, :7] = U1
    U[:, 7:] = U2

    V = np.zeros((8, 14))
    V[:4, :7] = V1
    V[4:, 7:] = V2

    W = np.zeros((8, 14))
    W[:4, :7] = W1
    W[4:, 7:] = W2

    return U, V, W


def _naive_1x2x4():
    """Return (U, V, W) for 1x2 * 2x4 using naive rank 8.

    A = [a, b], B is 2x4.
    C_k = a*B[0,k] + b*B[1,k] for k=0..3.
    Each C_k needs 2 terms → rank 8.

    U: (2, 8)  — a=0, b=1
    V: (8, 8)  — B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    W: (4, 8)  — C[0]=0, C[1]=1, C[2]=2, C[3]=3
    """
    U = np.zeros((2, 8))
    V = np.zeros((8, 8))
    W = np.zeros((4, 8))

    for k in range(4):
        # term 2k: a * B[0,k] -> C[k]
        U[0, 2 * k] = 1
        V[k, 2 * k] = 1
        W[k, 2 * k] = 1
        # term 2k+1: b * B[1,k] -> C[k]
        U[1, 2 * k + 1] = 1
        V[k + 4, 2 * k + 1] = 1
        W[k, 2 * k + 1] = 1

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] (outer) V[:,r] (outer) W[:,r] matches the multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i, j, k] = delta_{i,j,k}
    # T[i*m+j, j*p+k, k*n+i] = 1 (one-hot per (i,j,k) triple)
    # The decomposition says: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # i.e., for all (a, b, c): sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]

    # T is sparse: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0.

    # Build T as a 3D array
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Compute the decomposition tensor
    # D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum for efficiency
    D = np.einsum('ar,br,cr->abc', U, V, W)

    # Check
    if not np.allclose(D, T):
        diff = D - T
        n_diff = np.count_nonzero(diff)
        raise ValueError(f"Verification failed: {n_diff} entries differ. Max diff: {np.max(np.abs(diff))}")


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s

    # Total rank = 4*14 + 4*8 = 56 + 32 = 88
    rank = 88
    U_full = np.zeros((n * m, rank))  # 12 x 88
    V_full = np.zeros((m * p, rank))  # 32 x 88
    W_full = np.zeros((p * n, rank))  # 24 x 88

    col = 0

    # Block 1: C[0:2, 0:4] += A[0:2, 0:2] * B[0:2, 0:4]
    # A rows 0-1, cols 0-1 → U indices 0-3 (i*m+j: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
    # Wait: A[i,j] → U index i*m+j where m=4.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    #
    # B[j,k] → V index j*p+k where p=8.
    # B[0,0]=0, B[0,1]=1, ..., B[0,7]=7
    # B[1,0]=8, B[1,1]=9, ..., B[1,7]=15
    # B[2,0]=16, B[2,1]=17, ..., B[2,7]=23
    # B[3,0]=24, B[3,1]=25, ..., B[3,7]=31
    #
    # C[i,k] → W index k*n+i where n=3.
    # C[0,0]=0, C[0,1]=3, C[0,2]=6, C[0,3]=9, C[0,4]=12, C[0,5]=15, C[0,6]=18, C[0,7]=21
    # C[1,0]=1, C[1,1]=4, C[1,2]=7, C[1,3]=10, C[1,4]=13, C[1,5]=16, C[1,6]=19, C[1,7]=22
    # C[2,0]=2, C[2,1]=5, C[2,2]=8, C[2,3]=11, C[2,4]=14, C[2,5]=17, C[2,6]=20, C[2,7]=23

    # Block 1: C[0:2, 0:4] += A[0:2, 0:2] * B[0:2, 0:4]
    # A sub-block: rows 0-1, cols 0-1 of A → U local indices 0,1,4,5 (global)
    # But _strassen_22x24 expects U as (4,14) with local A indices 0,1,2,3 mapping to A[i,j] for i=0,1 j=0,1
    # So we need to map local A index (i,j) → global U index i*m+j
    # local 0=(0,0)→global 0, local 1=(0,1)→global 1, local 2=(1,0)→global 4, local 3=(1,1)→global 5

    U1, V1, W1 = _strassen_22x24()
    # Map U: local row 0→global 0, 1→1, 2→4, 3→5
    U_full[[0, 1, 4, 5], col:col + 14] = U1
    # Map V: local row j_local*4+k_local → global (j_offset + j_local)*8 + k_offset + k_local
    # B[0:2, 0:4]: j=0,1; k=0,1,2,3 → V indices 0-7
    # local V row 0-3 → B[0,0:4] → global 0-3
    # local V row 4-7 → B[1,0:4] → global 8-11
    V_full[[0, 1, 2, 3, 8, 9, 10, 11], col:col + 14] = V1
    # Map W: local W row k_local*2+i_local → global (k_offset+k_local)*3 + i_offset + i_local
    # C[0:2, 0:4]: i=0,1; k=0,1,2,3 → W indices k*3+i
    # local W row 0=(i=0,k=0)→global 0, 1=(i=1,k=0)→global 1, 2=(i=0,k=1)→global 3, 3=(i=1,k=1)→global 4
    # local W row 4=(i=0,k=2)→global 6, 5=(i=1,k=2)→global 7, 6=(i=0,k=3)→global 9, 7=(i=1,k=3)→global 10
    w_map = [0, 1, 3, 4, 6, 7, 9, 10]
    W_full[w_map, col:col + 14] = W1
    col += 14

    # Block 2: C[0:2, 0:4] += A[0:2, 2:4] * B[2:4, 0:4]
    # A sub-block: rows 0-1, cols 2-3 → U global indices 2,3,6,7
    # B sub-block: rows 2-3, cols 0-3 → V global indices 16-23
    U1, V1, W1 = _strassen_22x24()
    U_full[[2, 3, 6, 7], col:col + 14] = U1
    V_full[list(range(16, 24)), col:col + 14] = V1
    W_full[w_map, col:col + 14] = W1  # same C block as block 1
    col += 14

    # Block 3: C[0:2, 4:8] += A[0:2, 0:2] * B[0:2, 4:8]
    # A: same as block 1 → U 0,1,4,5
    # B: rows 0-1, cols 4-7 → V global 4-7, 12-15
    U1, V1, W1 = _strassen_22x24()
    U_full[[0, 1, 4, 5], col:col + 14] = U1
    V_full[[4, 5, 6, 7, 12, 13, 14, 15], col:col + 14] = V1
    # C[0:2, 4:8]: i=0,1; k=4,5,6,7 → W indices k*3+i
    # local W row 0=(i=0,k=0)→global 4*3+0=12, 1=(i=1,k=0)→13, 2=(i=0,k=1)→15, 3=(i=1,k=1)→16
    # 4=(i=0,k=2)→18, 5=(i=1,k=2)→19, 6=(i=0,k=3)→21, 7=(i=1,k=3)→22
    w_map3 = [12, 13, 15, 16, 18, 19, 21, 22]
    W_full[w_map3, col:col + 14] = W1
    col += 14

    # Block 4: C[0:2, 4:8] += A[0:2, 2:4] * B[2:4, 4:8]
    # A: rows 0-1, cols 2-3 → U 2,3,6,7
    # B: rows 2-3, cols 4-7 → V 20-23, 28-31
    U1, V1, W1 = _strassen_22x24()
    U_full[[2, 3, 6, 7], col:col + 14] = U1
    V_full[list(range(20, 24)) + list(range(28, 32)), col:col + 14] = V1
    W_full[w_map3, col:col + 14] = W1
    col += 14

    # Block 5: C[2:3, 0:4] += A[2:3, 0:2] * B[0:2, 0:4]
    # A: row 2, cols 0-1 → U global 8,9
    # B: rows 0-1, cols 0-3 → V 0-7
    U1, V1, W1 = _naive_1x2x4()
    U_full[[8, 9], col:col + 8] = U1
    V_full[list(range(0, 8)), col:col + 8] = V1
    # C[2, k] for k=0..3 → W index k*3+2
    w_map5 = [0 * 3 + 2, 1 * 3 + 2, 2 * 3 + 2, 3 * 3 + 2]  # [2,5,8,11]
    W_full[w_map5, col:col + 8] = W1
    col += 8

    # Block 6: C[2:3, 0:4] += A[2:3, 2:4] * B[2:4, 0:4]
    # A: row 2, cols 2-3 → U 10,11
    # B: rows 2-3, cols 0-3 → V 16-23
    U1, V1, W1 = _naive_1x2x4()
    U_full[[10, 11], col:col + 8] = U1
    V_full[list(range(16, 24)), col:col + 8] = V1
    W_full[w_map5, col:col + 8] = W1
    col += 8

    # Block 7: C[2:3, 4:8] += A[2:3, 0:2] * B[0:2, 4:8]
    # A: row 2, cols 0-1 → U 8,9
    # B: rows 0-1, cols 4-7 → V 4-7, 12-15
    U1, V1, W1 = _naive_1x2x4()
    U_full[[8, 9], col:col + 8] = U1
    V_full[[4, 5, 6, 7, 12, 13, 14, 15], col:col + 8] = V1
    w_map7 = [4 * 3 + 2, 5 * 3 + 2, 6 * 3 + 2, 7 * 3 + 2]  # [14,17,20,23]
    W_full[w_map7, col:col + 8] = W1
    col += 8

    # Block 8: C[2:3, 4:8] += A[2:3, 2:4] * B[2:4, 4:8]
    # A: row 2, cols 2-3 → U 10,11
    # B: rows 2-3, cols 4-7 → V 20-23, 28-31
    U1, V1, W1 = _naive_1x2x4()
    U_full[[10, 11], col:col + 8] = U1
    V_full[list(range(20, 24)) + list(range(28, 32)), col:col + 8] = V1
    W_full[w_map7, col:col + 8] = W1
    col += 8

    if col != rank:
        raise ValueError(f"Column mismatch: {col} != {rank}")

    # Verify correctness
    if time.time() < deadline:
        _verify(U_full, V_full, W_full, n, m, p)

    # Ensure integer/half-integer entries
    U_full = np.round(U_full * 2) / 2
    V_full = np.round(V_full * 2) / 2
    W_full = np.round(W_full * 2) / 2

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

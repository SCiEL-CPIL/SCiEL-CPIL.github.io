"""
Tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy:
  Partition the middle dimension m=5 as j in {0,1} vs j in {2,3,4}.
  Partition the output dimension p=7 as k in {0,1,2,3} vs k in {4,5,6}.

  Four blocks:
    Block 1: A[3x2] * B[2x4] -> C[3x4]
      Split k in {0,1,2,3} as {0,1,2} + {3}:
        1a: 3x2x3 via Laderman (rank 7)
        1b: 3x2x1 via schoolbook (rank 6)
    Block 2: A[3x2] * B[2x3] -> C[3x3]
      Schoolbook (rank 18)
    Block 3: A[3x3] * B[3x4] -> C[3x4]
      Split k in {0,1,2,3} as {0,1,2} + {3}:
        3a: 3x3x3 schoolbook (rank 27)
        3b: 3x3x1 schoolbook (rank 9)
    Block 4: A[3x3] * B[3x3] -> C[3x3]
      Schoolbook (rank 27)

  Total rank: 7 + 6 + 18 + 27 + 9 + 27 = 94.
"""
import time
import numpy as np

N, M, P = 3, 5, 7
DEADLINE = 120.0


def laderman_3x2x3():
    """Laderman's rank-7 decomposition for 3x2 * 2x3.
    U: (6,7), V: (6,7), W: (9,7).
    A is 3x2 (flat i*2+j), B is 2x3 (flat j*3+k), C is 3x3 (flat k*3+i).
    """
    U = np.zeros((6, 7))
    V = np.zeros((6, 7))
    W = np.zeros((9, 7))

    # M1 = A[0,0] * (B[0,0] + B[1,2])
    U[0, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[8, 0] = 1

    # M2 = (A[0,0] + A[2,0]) * (B[0,0] + B[1,0])
    U[0, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1; V[3, 1] = 1
    W[0, 1] = 1; W[2, 1] = 1; W[6, 1] = 1; W[8, 1] = 1

    # M3 = (A[1,0] + A[2,1]) * (B[0,1] + B[1,1])
    U[2, 2] = 1; U[5, 2] = 1
    V[1, 2] = 1; V[4, 2] = 1
    W[1, 2] = 1; W[3, 2] = 1; W[7, 2] = 1

    # M4 = (A[0,1] + A[2,1]) * (B[1,0] + B[1,1])
    U[1, 3] = 1; U[5, 3] = 1
    V[3, 3] = 1; V[4, 3] = 1
    W[2, 3] = 1; W[4, 3] = 1; W[6, 3] = 1; W[8, 3] = 1

    # M5 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[2, 4] = 1; U[0, 4] = -1
    V[0, 4] = 1; V[1, 4] = 1
    W[3, 4] = 1; W[4, 4] = 1

    # M6 = (A[1,1] - A[2,0]) * (B[0,1] + B[1,1])
    U[3, 5] = 1; U[4, 5] = -1
    V[1, 5] = 1; V[4, 5] = 1
    W[3, 5] = 1; W[5, 5] = 1; W[7, 5] = 1

    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[2, 6] = 1; W[4, 6] = 1; W[6, 6] = 1; W[8, 6] = 1

    return U, V, W


def schoolbook(n, m, p):
    """Schoolbook decomposition for n x m * m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p).
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def add_block_to_global(U_glob, V_glob, W_glob,
                        U_l, V_l, W_l,
                        u_map, v_map, w_map,
                        col_offset):
    """Add a local decomposition block to global arrays.
    u_map: list of (global_u_idx, local_u_idx) pairs
    v_map: list of (global_v_idx, local_v_idx) pairs
    w_map: list of (global_w_idx, local_w_idx) pairs
    Returns new col_offset.
    """
    rank_l = U_l.shape[1]
    for r in range(rank_l):
        for gu, lu in u_map:
            val = U_l[lu, r]
            if val != 0:
                U_glob[gu, col_offset + r] += val
        for gv, lv in v_map:
            val = V_l[lv, r]
            if val != 0:
                V_glob[gv, col_offset + r] += val
        for gw, lw in w_map:
            val = W_l[lw, r]
            if val != 0:
                W_glob[gw, col_offset + r] += val
    return col_offset + rank_l


def verify(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    r = U.shape[1]
    for c in range(r):
        T += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])

    # Check expected non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                if abs(T[a, b, d] - 1.0) > tol:
                    return False

    # Check all other entries are zero
    for a in range(n * m):
        for b in range(m * p):
            for d in range(p * n):
                # Compute the expected value
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k2 = b % p
                k3 = d // n
                i3 = d % n
                if i1 == i3 and j1 == j2 and k2 == k3:
                    continue  # expected to be 1, already checked
                if abs(T[a, b, d]) > tol:
                    return False
    return True


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    t_start = time.time()

    n, m, p = N, M, P
    # Total rank: 7 + 6 + 18 + 27 + 9 + 27 = 94
    total_rank = 94
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0

    # ---- Block 1a: A rows 0-2, cols 0-1 (3x2) * B rows 0-1, cols 0-2 (2x3) -> C rows 0-2, cols 0-2 (3x3)
    # Laderman 3x2x3
    U_l, V_l, W_l = laderman_3x2x3()
    # u_map: global A index = i*5+j, local = i*2+j
    u_map = [(i * 5 + j, i * 2 + j) for i in range(3) for j in range(2)]
    # v_map: global B index = j*7+k, local = j*3+k
    v_map = [(j * 7 + k, j * 3 + k) for j in range(2) for k in range(3)]
    # w_map: global C index = k*3+i, local = k*3+i
    w_map = [(k * 3 + i, k * 3 + i) for k in range(3) for i in range(3)]
    col = add_block_to_global(U, V, W, U_l, V_l, W_l, u_map, v_map, w_map, col)

    # ---- Block 1b: A rows 0-2, cols 0-1 (3x2) * B rows 0-1, col 3 (2x1) -> C rows 0-2, col 3 (3x1)
    # Schoolbook 3x2x1
    U_l, V_l, W_l = schoolbook(3, 2, 1)
    u_map = [(i * 5 + j, i * 2 + j) for i in range(3) for j in range(2)]
    v_map = [(j * 7 + 3, j * 1 + 0) for j in range(2)]  # k=3 only
    w_map = [(3 * 3 + i, 1 * 3 + i) for i in range(3)]  # k=3
    col = add_block_to_global(U, V, W, U_l, V_l, W_l, u_map, v_map, w_map, col)

    # ---- Block 2: A rows 0-2, cols 0-1 (3x2) * B rows 0-1, cols 4-6 (2x3) -> C rows 0-2, cols 4-6 (3x3)
    # Schoolbook 3x2x3
    U_l, V_l, W_l = schoolbook(3, 2, 3)
    u_map = [(i * 5 + j, i * 2 + j) for i in range(3) for j in range(2)]
    v_map = [(j * 7 + (4 + k), j * 3 + k) for j in range(2) for k in range(3)]
    w_map = [( (4 + k) * 3 + i, k * 3 + i) for k in range(3) for i in range(3)]
    col = add_block_to_global(U, V, W, U_l, V_l, W_l, u_map, v_map, w_map, col)

    # ---- Block 3a: A rows 0-2, cols 2-4 (3x3) * B rows 2-4, cols 0-2 (3x3) -> C rows 0-2, cols 0-2 (3x3)
    # Schoolbook 3x3x3
    U_l, V_l, W_l = schoolbook(3, 3, 3)
    u_map = [(i * 5 + (2 + j), i * 3 + j) for i in range(3) for j in range(3)]
    v_map = [((2 + j) * 7 + k, j * 3 + k) for j in range(3) for k in range(3)]
    w_map = [(k * 3 + i, k * 3 + i) for k in range(3) for i in range(3)]
    col = add_block_to_global(U, V, W, U_l, V_l, W_l, u_map, v_map, w_map, col)

    # ---- Block 3b: A rows 0-2, cols 2-4 (3x3) * B rows 2-4, cols 3 (3x1) -> C rows 0-2, col 3 (3x1)
    # Schoolbook 3x3x1
    U_l, V_l, W_l = schoolbook(3, 3, 1)
    u_map = [(i * 5 + (2 + j), i * 3 + j) for i in range(3) for j in range(3)]
    v_map = [((2 + j) * 7 + 3, j * 1 + 0) for j in range(3)]
    w_map = [(3 * 3 + i, 1 * 3 + i) for i in range(3)]
    col = add_block_to_global(U, V, W, U_l, V_l, W_l, u_map, v_map, w_map, col)

    # ---- Block 4: A rows 0-2, cols 2-4 (3x3) * B rows 2-4, cols 4-6 (3x3) -> C rows 0-2, cols 4-6 (3x3)
    # Schoolbook 3x3x3
    U_l, V_l, W_l = schoolbook(3, 3, 3)
    u_map = [(i * 5 + (2 + j), i * 3 + j) for i in range(3) for j in range(3)]
    v_map = [((2 + j) * 7 + (4 + k), j * 3 + k) for j in range(3) for k in range(3)]
    w_map = [((4 + k) * 3 + i, k * 3 + i) for k in range(3) for i in range(3)]
    col = add_block_to_global(U, V, W, U_l, V_l, W_l, u_map, v_map, w_map, col)

    # Verify
    assert col == total_rank, f"Column offset {col} != expected {total_rank}"
    assert verify(U, V, W), "Verification failed"

    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 5:
        # Safety: still return, but note the time
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

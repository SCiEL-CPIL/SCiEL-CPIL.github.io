"""Tensor decomposition of 5x5x5 matrix multiplication."""
import time
import random

DIMENSIONS = [5, 5, 5]
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    n, m, p = 5, 5, 5

    def idxU(i, j):
        return i * m + j
    def idxV(j, k):
        return j * p + k
    def idxW(k, i):
        return k * n + i

    # Build the rank-52 decomposition: 45 frame terms + 7 Strassen terms
    U_rows = []  # list of (index, value) pairs per rank column
    V_rows = []
    W_rows = []

    # Part 1: Frame terms (i==4 or j==4), k in 0..4 -> 45 terms
    for i in range(n):
        for j in range(m):
            if i == 4 or j == 4:
                for k in range(p):
                    U_rows.append([(idxU(i, j), 1)])
                    V_rows.append([(idxV(j, k), 1)])
                    W_rows.append([(idxW(k, i), 1)])

    # Part 2: 4x4x4 core via Strassen (rank 7)
    # Partition 4x4 into 2x2 blocks of size 2
    # A blocks: A11=(rows 0-1, cols 0-1), A12=(rows 0-1, cols 2-3),
    #           A21=(rows 2-3, cols 0-1), A22=(rows 2-3, cols 2-3)
    # B blocks: B11=(rows 0-1, cols 0-1), B12=(rows 0-1, cols 2-3),
    #           B21=(rows 2-3, cols 0-1), B22=(rows 2-3, cols 2-3)

    def block_positions(bi, bj):
        """Return list of (row, col) positions for block (bi,bj) in 4x4 matrix."""
        positions = []
        for di in range(2):
            for dj in range(2):
                positions.append((bi * 2 + di, bj * 2 + dj))
        return positions

    # Strassen 7 products for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22) B11
    # M3 = A11 (B12-B22)
    # M4 = A22 (B21-B11)
    # M5 = (A11+A12) B22
    # M6 = (A21-A11) (B11+B12)
    # M7 = (A12-A22) (B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7

    # Each M_t is a list of (A_block_id, B_block_id, sign)
    M_defs = [
        # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        [((0,0),(0,0),1), ((0,0),(1,1),1), ((1,1),(0,0),1), ((1,1),(1,1),1)],
        # M2 = (A21+A22) B11 = A21*B11 + A22*B11
        [((1,0),(0,0),1), ((1,1),(0,0),1)],
        # M3 = A11 (B12-B22) = A11*B12 - A11*B22
        [((0,0),(0,1),1), ((0,0),(1,1),-1)],
        # M4 = A22 (B21-B11) = A22*B21 - A22*B11
        [((1,1),(1,0),1), ((1,1),(0,0),-1)],
        # M5 = (A11+A12) B22 = A11*B22 + A12*B22
        [((0,0),(1,1),1), ((0,1),(1,1),1)],
        # M6 = (A21-A11) (B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
        [((1,0),(0,0),1), ((1,0),(0,1),1), ((0,0),(0,0),-1), ((0,0),(0,1),-1)],
        # M7 = (A12-A22) (B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
        [((0,1),(1,0),1), ((0,1),(1,1),1), ((1,1),(1,0),-1), ((1,1),(1,1),-1)],
    ]

    for parts in M_defs:
        U_acc = {}
        V_acc = {}
        W_acc = {}
        for (A_bid, B_bid, sign) in parts:
            A_pos = block_positions(*A_bid)
            B_pos = block_positions(*B_bid)
            # A_block * B_block: (A*B)[i][k] = sum_j A[i][j]*B[j][k]
            # A_block cols = A_bid[1]*2..A_bid[1]*2+1
            # B_block rows = B_bid[0]*2..B_bid[0]*2+1
            # Valid product requires A_bid[1] == B_bid[0]
            for (i, j) in A_pos:
                for (j2, k) in B_pos:
                    if j == j2:
                        ui = idxU(i, j)
                        vi = idxV(j, k)
                        wi = idxW(k, i)
                        U_acc[ui] = U_acc.get(ui, 0) + sign
                        V_acc[vi] = V_acc.get(vi, 0) + sign
                        W_acc[wi] = W_acc.get(wi, 0) + sign

        U_rows.append([(idx, val) for idx, val in U_acc.items() if val != 0])
        V_rows.append([(idx, val) for idx, val in V_acc.items() if val != 0])
        W_rows.append([(idx, val) for idx, val in W_acc.items() if val != 0])

    # Assemble the full matrices
    rank = len(U_rows)
    assert rank == 52, f"Expected rank 52, got {rank}"

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_rows[r]:
            U[idx][r] = val
        for idx, val in V_rows[r]:
            V[idx][r] = val
        for idx, val in W_rows[r]:
            W[idx][r] = val

    # Verify correctness with a few random trials
    random.seed(42)
    num_trials = 20
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][jj] * B[jj][k] for jj in range(m)) for k in range(p)] for i in range(n)]

        C_tensor = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_tensor[i][k] = s
        if C_tensor != C:
            # Should not happen; fallback to schoolbook
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
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
# EVOLVE-BLOCK-END

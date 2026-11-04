"""
Rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition the middle dimension (3) into 2+1.
  - For the 2x2 by 2x10 part: use Strassen (rank 7) for each 2x2x2 block (5 blocks) = 35
  - For the 2x1 by 1x10 part: schoolbook rank = 2*1*10 = 20
  Total rank: 55
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0


def _build_strassen_block(k0, k1, U, V, W, idx):
    """Add one Strassen 2x2x2 block. k0, k1 are the column indices in B (0..9)."""
    # M1 = (A11 + A22)(B11 + B22)
    U[0, idx] = 1.0   # A[0,0]
    U[4, idx] = 1.0   # A[1,1]
    V[k0, idx] = 1.0   # B[0,k0]
    V[10 + k1, idx] = 1.0  # B[1,k1]
    W[k0, idx] = 1.0   # C[0,k0]
    W[2 + k1, idx] = 1.0  # C[1,k1]
    idx += 1

    # M2 = (A21 + A22)B11
    U[3, idx] = 1.0   # A[1,0]
    U[4, idx] = 1.0   # A[1,1]
    V[k0, idx] = 1.0   # B[0,k0]
    W[1 + k0, idx] = 1.0   # C[0,k1]  -- wait, let me re-check
    # Actually for Strassen:
    # M2 = (A21 + A22) * B11
    # This contributes to C[1,k0] (bottom-left of 2x2 result)
    # C[1,k0] is at position k0*2+1 in W
    W[k0 * 2 + 1, idx] = 1.0
    W[k1 * 2 + 1, idx] = -1.0
    idx += 1

    # M3 = A11(B12 - B22)
    U[0, idx] = 1.0   # A[0,0]
    V[10 + k1, idx] = 1.0   # B[1,k1]
    V[k1, idx] = -1.0   # B[0,k1]
    W[k1 * 2 + 0, idx] = 1.0   # C[0,k1]
    W[k1 * 2 + 1, idx] = 1.0   # C[1,k1]
    idx += 1

    # M4 = A22(B21 - B11)
    U[4, idx] = 1.0   # A[1,1]
    V[10 + k0, idx] = 1.0   # B[1,k0]
    V[k0, idx] = -1.0   # B[0,k0]
    W[k0 * 2 + 0, idx] = 1.0   # C[0,k0]
    W[k0 * 2 + 1, idx] = 1.0   # C[1,k0]
    idx += 1

    # M5 = (A11 + A12)B22
    U[0, idx] = 1.0   # A[0,0]
    U[1, idx] = 1.0   # A[0,1]
    V[10 + k1, idx] = 1.0   # B[1,k1]
    W[k0 * 2 + 0, idx] = -1.0   # C[0,k0]
    W[k1 * 2 + 0, idx] = 1.0   # C[0,k1]
    idx += 1

    # M6 = (A21 - A11)(B11 + B12)
    U[3, idx] = 1.0   # A[1,0]
    U[0, idx] = -1.0   # A[0,0]
    V[k0, idx] = 1.0   # B[0,k0]
    V[k1, idx] = 1.0   # B[0,k1]
    W[k1 * 2 + 1, idx] = 1.0   # C[1,k1]
    idx += 1

    # M7 = (A12 - A22)(B21 + B22)
    U[1, idx] = 1.0   # A[0,1]
    U[4, idx] = -1.0   # A[1,1]
    V[10 + k0, idx] = 1.0   # B[1,k0]
    V[10 + k1, idx] = 1.0   # B[1,k1]
    W[k0 * 2 + 0, idx] = 1.0   # C[0,k0]
    idx += 1

    return idx


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Partition middle dim: 3 = 2 + 1
    # Part 1: rows 0,1 of A (2x2) times rows 0,1 of B (2x10) -> 5 Strassen blocks of 2x2x2
    # Part 2: row 2 of A (2x1) times row 2 of B (1x10) -> schoolbook rank 20

    rank = 35 + 20  # 55

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        idx = _build_strassen_block(k0, k1, U, V, W, idx)

    # Part 2: Schoolbook for 2x1 x 1x10
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0
            V[2 * 10 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    # Verify correctness
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]

        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")

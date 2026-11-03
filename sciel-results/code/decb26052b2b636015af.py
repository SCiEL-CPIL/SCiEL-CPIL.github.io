"""Matrix multiplication 3,3,6 rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct an exact decomposition for 3x3x6 matrix multiplication.

    Split the 6 columns of B into three groups of 2 columns. For each group use a
    decomposition of 3x3x2 (Strassen's 7 terms for the 2x2x2 core plus 10
    schoolbook terms for the remaining products). Total rank: 3 * 17 = 51.

    Index conventions (official trace convention):
      U index: i*m + j
      V index: j*p + k
      W index: k*n + i
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6

    # Build the 17-term decomposition for a 3x3x2 block.
    # Local indices: A[i,j] i,j in {0,1,2}; B[j,k] j in {0,1,2}, k in {0,1};
    # C[i,k] i in {0,1,2}, k in {0,1}.
    # Local U index: i*3 + j ; V index: j*2 + k ; W index: k*3 + i.

    U_block = []
    V_block = []
    W_block = []

    # Strassen 7 terms for 2x2x2 core.
    # m1 = (a00+a11)*(b00+b11)
    U_block.append([1,0,0, 1,0,0, 0,0,0])
    V_block.append([1,0, 1,0, 0,0])
    W_block.append([1,0,0, 0,1,0, 0,0,0])  # +c00 +c11

    # m2 = (a10+a11)*b00
    U_block.append([0,0,0, 1,1,0, 0,0,0])
    V_block.append([1,0, 0,0, 0,0])
    W_block.append([0,0,0, 0,1,0, 0,0,0])  # +c10

    # m3 = a00*(b01-b11)
    U_block.append([1,0,0, 0,0,0, 0,0,0])
    V_block.append([0,1, -1,0, 0,0])
    W_block.append([0,0,0, 1,0,0, 0,0,0])  # +c01 (and -c11 handled by sign in m1/m3 combination)

    # Actually, Strassen reconstruction:
    # c00 = m1 - m4 + m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m6 - m7
    # So each m contributes to specific C entries with specific signs.
    # m1: +c00, +c11
    # m2: +c10
    # m3: +c01, -c11
    # m4: -c10
    # m5: +c00, +c01
    # m6: +c11
    # m7: -c00, -c11

    # Let me redo the W factors correctly.

    U_block = []
    V_block = []
    W_block = []

    # m1 = (a00+a11)*(b00+b11) -> +c00, +c11
    U_block.append([1,0,0, 1,0,0, 0,0,0])
    V_block.append([1,0, 1,0, 0,0])
    W_block.append([1,0,0, 0,1,0, 0,0,0])

    # m2 = (a10+a11)*b00 -> +c10
    U_block.append([0,0,0, 1,1,0, 0,0,0])
    V_block.append([1,0, 0,0, 0,0])
    W_block.append([0,0,0, 0,1,0, 0,0,0])

    # m3 = a00*(b01-b11) -> +c01, -c11
    U_block.append([1,0,0, 0,0,0, 0,0,0])
    V_block.append([0,1, -1,0, 0,0])
    W_block.append([0,0,0, 1,0,0, 0,0,0])  # c01
    # Need -c11: add to W
    W_block[-1] = [0,0,0, 1,0,0, 0,-1,0]  # +c01 -c11

    # m4 = a11*(b10-b00) -> -c10
    U_block.append([0,0,0, 0,1,0, 0,0,0])
    V_block.append([-1,0, 1,0, 0,0])
    W_block.append([0,0,0, 0,1,0, 0,0,0])  # c10
    W_block[-1] = [0,0,0, 0,-1,0, 0,0,0]  # -c10

    # m5 = (a00+a01)*b11 -> +c00, +c01
    U_block.append([1,1,0, 0,0,0, 0,0,0])
    V_block.append([0,0, 0,1, 0,0])
    W_block.append([1,0,0, 0,0,0, 0,0,0])  # c00
    W_block[-1] = [1,0,0, 1,0,0, 0,0,0]  # +c00 +c01

    # m6 = (a10-a00)*(b00+b01) -> +c11
    U_block.append([-1,0,0, 1,0,0, 0,0,0])
    V_block.append([1,1, 0,0, 0,0])
    W_block.append([0,0,0, 0,0,0, 0,1,0])  # +c11

    # m7 = (a01-a11)*(b10+b11) -> -c00, -c11
    U_block.append([0,1,0, 0,-1,0, 0,0,0])
    V_block.append([0,0, 1,1, 0,0])
    W_block.append([1,0,0, 0,1,0, 0,0,0])  # c00, c11
    W_block[-1] = [-1,0,0, 0,-1,0, 0,0,0]  # -c00 -c11

    # 10 schoolbook terms for remaining products
    # a02*b20 -> c00
    U_block.append([0,0,1, 0,0,0, 0,0,0]); V_block.append([0,0, 0,0, 1,0]); W_block.append([1,0,0, 0,0,0, 0,0,0])
    # a02*b21 -> c01
    U_block.append([0,0,1, 0,0,0, 0,0,0]); V_block.append([0,0, 0,0, 0,1]); W_block.append([0,0,0, 1,0,0, 0,0,0])
    # a12*b20 -> c10
    U_block.append([0,0,0, 0,0,1, 0,0,0]); V_block.append([0,0, 0,0, 1,0]); W_block.append([0,0,0, 0,1,0, 0,0,0])
    # a12*b21 -> c11
    U_block.append([0,0,0, 0,0,1, 0,0,0]); V_block.append([0,0, 0,0, 0,1]); W_block.append([0,0,0, 0,0,0, 0,1,0])
    # a20*b00 -> c20
    U_block.append([0,0,0, 0,0,0, 1,0,0]); V_block.append([1,0, 0,0, 0,0]); W_block.append([0,0,0, 0,0,0, 1,0,0])
    # a20*b01 -> c21
    U_block.append([0,0,0, 0,0,0, 1,0,0]); V_block.append([0,1, 0,0, 0,0]); W_block.append([0,0,0, 0,0,0, 0,0,1])
    # a21*b10 -> c20
    U_block.append([0,0,0, 0,0,0, 0,1,0]); V_block.append([0,0, 1,0, 0,0]); W_block.append([0,0,0, 0,0,0, 1,0,0])
    # a21*b11 -> c21
    U_block.append([0,0,0, 0,0,0, 0,1,0]); V_block.append([0,0, 0,1, 0,0]); W_block.append([0,0,0, 0,0,0, 0,0,1])
    # a22*b20 -> c20
    U_block.append([0,0,0, 0,0,0, 0,0,1]); V_block.append([0,0, 0,0, 1,0]); W_block.append([0,0,0, 0,0,0, 1,0,0])
    # a22*b21 -> c21
    U_block.append([0,0,0, 0,0,0, 0,0,1]); V_block.append([0,0, 0,0, 0,1]); W_block.append([0,0,0, 0,0,0, 0,0,1])

    block_rank = len(U_block)  # 17

    # Assemble global factors: 3 blocks, each of rank 17, total rank 51.
    total_rank = 3 * block_rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for s in range(3):
        base = s * block_rank
        for r in range(block_rank):
            col = base + r
            u = U_block[r]
            v = V_block[r]
            w = W_block[r]
            # U: same index i*3+j
            for idx in range(9):
                if u[idx] != 0:
                    U[idx][col] = u[idx]
            # V: local j*2+k -> global j*6 + (2*s + k)
            for j in range(3):
                for k in range(2):
                    local_idx = j * 2 + k
                    if v[local_idx] != 0:
                        global_idx = j * 6 + (2 * s + k)
                        V[global_idx][col] = v[local_idx]
            # W: local k*3+i -> global (2*s+k)*3 + i
            for k in range(2):
                for i in range(3):
                    local_idx = k * 3 + i
                    if w[local_idx] != 0:
                        global_idx = (2 * s + k) * 3 + i
                        W[global_idx][col] = w[local_idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

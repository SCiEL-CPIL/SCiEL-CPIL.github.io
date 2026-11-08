"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 118

    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]   # (8, 36)
    V = [[0] * rank for _ in range(m * p)]   # (20, 36)
    W = [[0] * rank for _ in range(p * n)]   # (10, 36)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    def strassen_block(y, z, w):
        """Apply 2x2x2 Strassen (rank 7).
        y = [(i0,j0),(i0,j1),(i1,j0),(i1,j1)]
        z = [(j0,k0),(j0,k1),(j1,k0),(j1,k1)]
        w = [(k0,i0),(k1,i0),(k0,i1),(k1,i1)]
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z

        # M1 = (y00+y11)*(z00+z11) -> C00 += M1, C11 += M1
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w[0])][col] += 1
        W[w_idx(*w[3])][col] += 1
        col += 1

        # M2 = (y10+y11)*z00 -> C10 += M2, C11 -= M2
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w[2])][col] += 1
        W[w_idx(*w[3])][col] -= 1
        col += 1

        # M3 = y00*(z01-z11) -> C01 += M3, C11 += M3
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z01)][col] = 1
        V[v_idx(*z11)][col] = -1
        W[w_idx(*w[1])][col] += 1
        W[w_idx(*w[3])][col] += 1
        col += 1

        # M4 = y11*(z10-z00) -> C00 += M4, C10 += M4
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z00)][col] = -1
        W[w_idx(*w[0])][col] += 1
        W[w_idx(*w[2])][col] += 1
        col += 1

        # M5 = (y00+y01)*z11 -> C00 -= M5, C01 += M5
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w[0])][col] -= 1
        W[w_idx(*w[1])][col] += 1
        col += 1

        # M6 = (y10-y00)*(z00+z01) -> C11 += M6
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y00)][col] = -1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w[3])][col] += 1
        col += 1

        # M7 = (y01-y11)*(z10+z11) -> C00 += M7
        U[u_idx(*y01)][col] = 1
        U[u_idx(*y11)][col] = -1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w[0])][col] += 1
        col += 1

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen_block(
        [(0,0),(0,1),(1,0),(1,1)],
        [(0,0),(0,1),(1,0),(1,1)],
        [(0,0),(1,0),(0,1),(1,1)]
    )

    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    strassen_block(
        [(0,2),(0,3),(1,2),(1,3)],
        [(2,0),(2,1),(3,0),(3,1)],
        [(0,0),(1,0),(0,1),(1,1)]
    )

    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    strassen_block(
        [(0,0),(0,1),(1,0),(1,1)],
        [(0,2),(0,3),(1,2),(1,3)],
        [(2,0),(3,0),(2,1),(3,1)]
    )

    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    strassen_block(
        [(0,2),(0,3),(1,2),(1,3)],
        [(2,2),(2,3),(3,2),(3,3)],
        [(2,0),(3,0),(2,1),(3,1)]
    )

    # Block 5: A[0:2,0:2] * B[0:2,4] (2x2 * 2x1, rank 4)
    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1

    # Block 6: A[0:2,2:4] * B[2:4,4] (2x2 * 2x1, rank 4)
    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

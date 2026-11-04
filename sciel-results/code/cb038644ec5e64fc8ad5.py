"""Matrix multiplication 2,4,6 - rank-minimal tensor decomposition using Strassen blocks."""
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = 2, 4, 6
    rank = 42

    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    def add_strassen(i0, j0, k0, off):
        """Add a Strassen 2x2x2 block: A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        # M1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
        c = off
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1

        # M2: (A10+A11)B00 -> C10 +=, C11 -=
        c = off + 1
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1

        # M3: A00(B01-B11) -> C01 +=, C11 +=
        c = off + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1

        # M4: A11(B10-B00) -> C00 +=, C10 +=
        c = off + 3
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1

        # M5: (A00+A01)B11 -> C00 -=, C01 +=
        c = off + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1

        # M6: (A10-A00)(B00+B01) -> C11 +=
        c = off + 5
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1

        # M7: (A01-A11)(B10+B11) -> C00 +=
        c = off + 6
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1

    # 6 blocks: A split into 2 row-groups (rows 0-1, but A is only 2 rows so one group of 2)
    # Actually A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x6: rows 0,1,2,3; cols 0,1,2,3,4,5
    # C is 2x6: rows 0,1; cols 0,1,2,3,4,5
    
    # Block structure: A_cols split {0,1},{2,3}; B_rows same; B_cols split {0,1},{2,3},{4,5}
    # 2 * 3 = 6 blocks, each 2x2x2
    
    add_strassen(0, 0, 0, 0)    # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    add_strassen(0, 0, 2, 7)    # A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    add_strassen(0, 0, 4, 14)   # A[:,0:2] * B[0:2,4:6] -> C[:,4:6]
    add_strassen(0, 2, 0, 21)   # A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    add_strassen(0, 2, 2, 28)   # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    add_strassen(0, 2, 4, 35)   # A[:,2:4] * B[2:4,4:6] -> C[:,4:6]

    if time.time() > deadline:
        pass  # Already computed, just return

    return {"U": U, "V": V, "W": W}

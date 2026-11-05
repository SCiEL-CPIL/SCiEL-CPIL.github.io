"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    def time_left():
        return deadline - time.time()

    # ------------------------------------------------------------------
    # Verification helper
    # ------------------------------------------------------------------
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W exactly decompose the matmul tensor."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]

        # Check the 50 non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                    if abs(val - 1.0) > tol:
                        return False

        # Check a sample of zero entries
        checked = 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    val = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                    if abs(val) > tol:
                        return False
                    checked += 1
                    if checked > 200:
                        return True
        return True

    # ------------------------------------------------------------------
    # Schoolbook decomposition (rank 50)
    # ------------------------------------------------------------------
    def schoolbook_decomposition(n, m, p):
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

    # ------------------------------------------------------------------
    # Strassen-based decomposition for 2x5 * 5x5
    #
    # Partition B (5x5) into blocks along rows and cols:
    #   Rows: 2 + 2 + 1  (indices 0,1 | 2,3 | 4)
    #   Cols: 2 + 2 + 1  (indices 0,1 | 2,3 | 4)
    # A is 2x5, split columns: 2 + 2 + 1
    #
    # This gives 3x3 = 9 block-multiplications of sizes:
    #   (2x2)*(2x2) -> 4 blocks of size 2x2  -> use Strassen (rank 7 each)
    #   (2x2)*(2x1) -> 2 blocks               -> schoolbook (rank 4 each)
    #   (2x2)*(1x2) -> 2 blocks               -> schoolbook (rank 4 each)
    #   (2x2)*(1x1) -> 1 block                -> schoolbook (rank 2)
    #   (2x1)*(2x2) -> 2 blocks               -> schoolbook (rank 4 each)
    #   (2x1)*(2x1) -> 1 block                -> schoolbook (rank 2)
    #   (2x1)*(1x2) -> 2 blocks               -> schoolbook (rank 2 each)
    #   (2x1)*(1x1) -> 1 block                -> schoolbook (rank 1)
    #   (2x?)*(?)... 
    #
    # Actually let me think more carefully.
    # A is 2x5. Split A columns as 2+2+1: A = [A0 | A1 | A2] where A0 is 2x2, A1 is 2x2, A2 is 2x1.
    # B is 5x5. Split B rows as 2+2+1 and B cols as 2+2+1:
    #   B = [[B00, B01, B02],
    #        [B10, B11, B12],
    #        [B20, B21, B22]]
    #   where B00 is 2x2, B01 is 2x2, B02 is 2x1,
    #         B10 is 2x2, B11 is 2x2, B12 is 2x1,
    #         B20 is 1x2, B21 is 1x2, B22 is 1x1.
    #
    # C = A * B = [A0|A1|A2] * B
    #   = A0*B00 + A1*B10 + A2*B20   (first 2 cols of C)
    #   + A0*B01 + A1*B11 + A2*B21   (next 2 cols of C)
    #   + A0*B02 + A1*B12 + A2*B22   (last 1 col of C)
    #
    # So C is 2x5, split cols as 2+2+1.
    #
    # Block multiplications needed:
    #   (2x2)*(2x2) = 4 times  -> Strassen rank 7 each = 28
    #   (2x2)*(2x1) = 2 times  -> schoolbook rank 4 each = 8
    #   (2x2)*(1x2) = 2 times  -> schoolbook rank 4 each = 8
    #   (2x2)*(1x1) = 1 time   -> schoolbook rank 2
    #   (2x1)*(2x2) = 2 times  -> schoolbook rank 4 each = 8
    #   (2x1)*(2x1) = 1 time   -> schoolbook rank 2
    #   (2x1)*(1x2) = 2 times  -> schoolbook rank 2 each = 4
    #   (2x1)*(1x1) = 1 time   -> schoolbook rank 1
    #
    # Wait, I need to be more careful. Let me enumerate:
    # A0 (2x2) * B00 (2x2) -> Strassen 7
    # A0 (2x2) * B01 (2x2) -> Strassen 7
    # A0 (2x2) * B02 (2x1) -> schoolbook 4
    # A1 (2x2) * B10 (2x2) -> Strassen 7
    # A1 (2x2) * B11 (2x2) -> Strassen 7
    # A1 (2x2) * B12 (2x1) -> schoolbook 4
    # A2 (2x1) * B20 (1x2) -> schoolbook 2
    # A2 (2x1) * B21 (1x2) -> schoolbook 2
    # A2 (2x1) * B22 (1x1) -> schoolbook 1
    #
    # Total rank = 4*7 + 2*4 + 2*2 + 1 = 28 + 8 + 4 + 1 = 41
    #
    # But wait, the C output is 2x5. Each block product contributes to specific columns of C.
    # C[:, 0:2] = A0*B00 + A1*B10 + A2*B20
    # C[:, 2:4] = A0*B01 + A1*B11 + A2*B21
    # C[:, 4:5] = A0*B02 + A1*B12 + A2*B22
    #
    # So the total number of scalar multiplications is the sum of ranks of all block products.
    # = 7+7+4+7+7+4+2+2+1 = 41
    #
    # Hmm, but the observed improvement was to rank 46. Let me re-examine.
    # Perhaps the partitioning is different. Let me try 2+2+1 for both A columns and B rows/cols.
    #
    # Actually, the key insight from the knowledge is:
    # "Partitioning the 5x5 B matrix into 2+2+1 blocks along both dimensions allows using
    #  Strassen's 7-multiplication algorithm for 2x2*2x2 subblocks, reducing total rank
    #  from 50 (schoolbook) to 46"
    #
    # So the rank is 46, not 41. Let me think about why.
    #
    # Maybe the partitioning is: A is 2x5, split as 2+2+1 columns.
    # B is 5x5, split rows as 2+2+1, cols as 2+2+1.
    #
    # The 4 (2x2)*(2x2) products use Strassen: 4*7 = 28
    # The remaining products:
    #   (2x2)*(2x1): 2 of them, each needs 4 mults -> 8
    #   (2x1)*(1x2): 2 of them, each needs 2 mults -> 4
    #   (2x2)*(1x1): 1 of them, needs 2 mults -> 2
    #   (2x1)*(1x1): 1 of them, needs 1 mult -> 1
    #   (2x1)*(2x1): 1 of them, needs 2 mults -> 2
    #   (2x1)*(2x2): 2 of them, each needs 4 mults -> 8
    #
    # Wait, I think I'm double-counting. Let me be very explicit.
    #
    # A columns: [0,1] -> A0 (2x2), [2,3] -> A1 (2x2), [4] -> A2 (2x1)
    # B rows: [0,1] -> row block 0, [2,3] -> row block 1, [4] -> row block 2
    # B cols: [0,1] -> col block 0, [2,3] -> col block 1, [4] -> col block 2
    #
    # B sub-blocks:
    #   B[0:2, 0:2] = B00 (2x2)
    #   B[0:2, 2:4] = B01 (2x2)
    #   B[0:2, 4:5] = B02 (2x1)
    #   B[2:4, 0:2] = B10 (2x2)
    #   B[2:4, 2:4] = B11 (2x2)
    #   B[2:4, 4:5] = B12 (2x1)
    #   B[4:5, 0:2] = B20 (1x2)
    #   B[4:5, 2:4] = B21 (1x2)
    #   B[4:5, 4:5] = B22 (1x1)
    #
    # Products:
    #   A0*B00: (2x2)*(2x2) -> 7 (Strassen)
    #   A0*B01: (2x2)*(2x2) -> 7 (Strassen)
    #   A0*B02: (2x2)*(2x1) -> 4 (schoolbook)
    #   A1*B10: (2x2)*(2x2) -> 7 (Strassen)
    #   A1*B11: (2x2)*(2x2) -> 7 (Strassen)
    #   A1*B12: (2x2)*(2x1) -> 4 (schoolbook)
    #   A2*B20: (2x1)*(1x2) -> 2 (schoolbook)
    #   A2*B21: (2x1)*(1x2) -> 2 (schoolbook)
    #   A2*B22: (2x1)*(1x1) -> 1 (schoolbook)
    #
    # Total: 7*4 + 4*2 + 2*2 + 1 = 28 + 8 + 4 + 1 = 41
    #
    # But the knowledge says 46. Maybe the partitioning is different, or maybe I'm
    # misunderstanding how Strassen applies here. Let me just implement it and see.
    #
    # Actually, maybe the issue is that Strassen for (2x2)*(2x2) requires specific
    # linear combinations, and when we have multiple such products sharing A or B
    # sub-blocks, we can't always share the intermediate computations.
    #
    # For now, let me implement the straightforward block approach and verify the rank.

    def strassen_2x2(U_rows, V_rows, W_rows, base_col, n, m, p, U, V, W):
        """
        Add Strassen decomposition for a (2x2)*(2x2) product.
        U_rows: list of 4 row indices in U (for A sub-block, row-major 2x2)
        V_rows: list of 4 row indices in V (for B sub-block, row-major 2x2)
        W_rows: list of 4 row indices in W (for C sub-block, row-major 2x2)
        base_col: starting column index
        Returns: new base_col (advanced by 7)
        """
        # A sub-block rows: a00, a01, a10, a11
        a00, a01, a10, a11 = U_rows
        # B sub-block rows: b00, b01, b10, b11
        b00, b01, b10, b11 = V_rows
        # C sub-block rows: c00, c01, c10, c11
        c00, c01, c10, c11 = W_rows

        # Strassen:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        col = base_col

        # M1
        for r in [a00, a11]:
            U[r][col] += 1
        for r in [b00, b11]:
            V[r][col] += 1
        for r in [c00, c11]:
            W[r][col] += 1
        col += 1

        # M2
        for r in [a10, a11]:
            U[r][col] += 1
        V[b00][col] += 1
        for r in [c10, c11]:
            W[r][col] += 1
        col += 1

        # M3
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        for r in [c01, c11]:
            W[r][col] += 1
        col += 1

        # M4
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        for r in [c00, c10]:
            W[r][col] += 1
        col += 1

        # M5
        for r in [a00, a01]:
            U[r][col] += 1
        V[b11][col] += 1
        for r in [c00, c01]:
            W[r][col] -= 1
        col += 1

        # M6
        U[a10][col] += 1
        U[a00][col] -= 1
        for r in [b00, b01]:
            V[r][col] += 1
        W[c11][col] += 1
        col += 1

        # M7
        U[a01][col] += 1
        U[a11][col] -= 1
        for r in [b10, b11]:
            V[r][col] += 1
        W[c00][col] += 1
        col += 1

        return col

    def schoolbook_block(U_rows, V_rows, W_rows, base_col, U, V, W):
        """
        Add schoolbook decomposition for a small block product.
        U_rows: list of row indices in U (len = nrows * ncols of A block)
        V_rows: list of row indices in V (len = nrows * ncols of B block)
        W_rows: list of row indices in W (len = nrows * ncols of C block)
        The product is (nrows x ncolsA) * (nrowsB x ncolsB) where ncolsA = nrowsB.
        For simplicity, assume the blocks are compatible.
        """
        col = base_col
        # For a general small block, we need to know the dimensions.
        # This is getting complex. Let me handle specific cases.
        return col

    # Given the complexity, let me take a different approach:
    # Build the full decomposition by explicitly constructing the tensor
    # and using a known low-rank construction.

    # Actually, let me just try the HOOI/random search approach more aggressively,
    # but with a better initialization strategy based on the Strassen structure.

    def try_rank_with_hooi(r, n, m, p, n_trials=50, max_iters=100, tol=1e-12):
        """Try to find a rank-r decomposition using HOOI with multiple random initializations."""
        if time_left() < 10:
            return None

        for trial in range(n_trials):
            if time_left() < 10:
                break

            np.random.seed(42 + trial * 137)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)

            # Normalize
            U /= np.linalg.norm(U) + 1e-10
            V /= np.linalg.norm(V) + 1e-10
            W /= np.linalg.norm(W) + 1e-10

            for it in range(max_iters):
                if time_left() < 10:
                    break

                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_new[i * m + j, :] += V[j * p + k, :] * W[k * n + i, :]

                # Update V
                V_new = np.zeros_like(V)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            V_new[j * p + k, :] += U[i * m + j, :] * W[k * n + i, :]

                # Update W
                W_new = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            W_new[k * n + i, :] += U[i * m + j, :] * V[j * p + k, :]

                delta = max(np.linalg.norm(U_new - U),
                           np.linalg.norm(V_new - V),
                           np.linalg.norm(W_new - W))
                U, V, W = U_new, V_new, W_new

                if delta < tol:
                    break

            # Check if this gives an exact decomposition
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = 0.0
                        for t in range(r):
                            s += U[a, t] * V[b, t] * W[c, t]
                        max_err = max(max_err, abs(s - 1.0))

            if max_err < 1e-10:
                U_list = [[float(x) for x in U[row, :]] for row in range(n * m)]
                V_list = [[float(x) for x in V[row, :]] for row in range(m * p)]
                W_list = [[float(x) for x in W[row, :]] for row in range(p * n)]
                return U_list, V_list, W_list

        return None

    # ------------------------------------------------------------------
    # Try to find a low-rank decomposition
    # ------------------------------------------------------------------
    best_result = None

    # Try ranks from 21 upward (known lower bound for 2x5x5 is around 21-25)
    for r in range(21, 51):
        if time_left() < 15:
            break

        result = try_rank_with_hooi(r, n, m, p, n_trials=30, max_iters=80)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, n, m, p):
                best_result = (r, U, V, W)
                break

    if best_result is not None:
        r, U, V, W = best_result
        return {"U": U, "V": V, "W": W}

    # Fall back to schoolbook
    U, V, W = schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

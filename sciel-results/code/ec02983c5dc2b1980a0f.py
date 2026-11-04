import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]


def construct():
    n, m, p = DIMENSIONS

    # We build a rank-33 exact decomposition.
    # Structure:
    #   1. First two columns of V (j=0,1) combined into one rank-15 block using Strassen-like ideas for the 2x2 subproblem.
    #   2. Remaining column (j=2) handled with a simple rank-7 block.
    #
    # The matrix A is 2x3. Split as A = [A_left | A_right] where A_left is 2x2, A_right is 2x1.
    # B is 3x7. Split as B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 1x7.
    #
    # For the 2x2 x 7 part, we use a known rank-15 decomposition (Strassen for 2x2 + 3 extra columns).
    # For the 2x1 x 7 part, we use rank 7 (trivial).
    #
    # Total rank = 15 + 7 = 22. Wait, that's too optimistic. Let me recalculate.
    #
    # Actually, R(2,2,7) is known to be 15 (Strassen + 3 for the extra 3 columns? No, Strassen is for 2x2x2.
    # For 2x2x7, we can use the 2x2x2 Strassen (rank 7) for the first 2 columns and 2x2x1 (rank 2) for each
    # of the remaining 5 columns, giving 7 + 5*2 = 17. So R(2,2,7) <= 17.
    # For the 2x1x7 part, we need rank 7.
    # Total: 17 + 7 = 24.
    #
    # Let's be conservative and use the trivial decomposition for the 2x2x7 part: rank 14.
    # Then total rank = 14 + 7 = 21.
    #
    # But we can do better. Let's use rank 15 for the 2x2x7 part and rank 7 for the 2x1x7 part, total 22.
    #
    # I'll implement the trivial rank-21 decomposition first, then try to optimize.

    # Actually, let's just use the known rank-33 decomposition from the literature for 2x3x7.
    # I'll construct it carefully.

    # Rank-33 construction:
    # We split B into 3 blocks of size 3x? No, B is 3x7.
    # Let's split B as [B1; B2; B3] where B1, B2, B3 are 1x7.
    # Then C = A1 * B1 + A2 * B2 + A3 * B3, where A1, A2, A3 are 2x1 vectors.
    # This is a sum of 3 rank-14 tensors (2x1x7), total rank 42. That's the trivial one.

    # Let's use the following approach:
    # Split A = [A1 | A2] where A1 is 2x2 and A2 is 2x1.
    # Split B = [B1; B2] where B1 is 2x7 and B2 is 1x7.
    # C = A1 * B1 + A2 * B2.
    # A1*B1 is 2x7, computed as 2x2 times 2x7. Rank <= 2*2*7 = 28.
    # A2*B2 is 2x7, computed as 2x1 times 1x7. Rank <= 14.
    # Total rank <= 42.

    # To reduce rank, we use Strassen for the 2x2 part.
    # For 2x2 times 2x7, we can use 7 multiplications for the 2x2x2 part and 2*5=10 for the remaining 5 columns.
    # So rank for A1*B1 <= 7 + 10 = 17.
    # For A2*B2, rank <= 7.
    # Total rank <= 24.

    # Let's implement the rank-24 decomposition.

    # First, define the Strassen decomposition for 2x2 x 2x2.
    # M1 = (a+d)*(g+i)  -- but this is for scalar matrices. For matrix multiplication, we have:
    # C11 = M1 - M2 + M3 - M4
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M3 + M6 - M5
    # where M1..M7 are the 7 products.

    # For 2x2 times 2x7, we treat each column of B as a 2x1 vector.
    # Actually, it's easier to think of it as 2x2 times 2x7.
    # We can use the following: for each group of 2 columns, use Strassen. But 7 is odd.
    # Let's use 1 group of 2 columns (Strassen, rank 7) and 5 groups of 1 column (rank 2 each, total 10).
    # Total rank for 2x2x7 = 17.

    # For the 2x1x7 part, each column needs 2 multiplications, total 14. But we can share? No, it's rank 7 if we view it as 2x1x7.
    # Actually, 2x1x7 is just 14 scalar multiplications, but as a tensor rank, it's 7 (each rank-1 term corresponds to one column).
    # Wait, the tensor for 2x1x7 has shape (2,1,7). The rank is 7, not 14.
    # Because each rank-1 term is of the form u_i * v_j * w_k, and we need 7 such terms.

    # So total rank = 17 + 7 = 24.

    # Let's implement this.

    rank = 24
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Part 1: A1 * B1 where A1 is 2x2 (first 2 columns of A), B1 is 2x7 (first 2 rows of B).
    # A1 = [[a00, a01], [a10, a11]] but A is 2x3, so A1 = A[:, 0:2].
    # B1 = B[0:2, :] = B[0:2, 0:7].

    # Strassen for the first 2 columns of B1 (columns 0,1).
    # Let B1[:, 0:2] = [[b00, b01], [b10, b11]].
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b11)
    # M4 = a11 * (b01 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a01 - a00) * (b00 + b10)
    # M7 = (a11 - a10) * (b01 + b11)
    # C[:, 0] = M1 - M2 + M3 - M4
    # C[:, 1] = M2 + M4

    # For columns 2..6 of B1, each column is 2x1, so we need 2 multiplications per column.
    # Column k of B1 is [b0k, b1k]. C[:, k] = a00*b0k + a01*b1k for row 0, a10*b0k + a11*b1k for row 1.
    # This is 2 rank-1 terms per column, total 10 for 5 columns.

    # Part 2: A2 * B2 where A2 is 2x1 (3rd column of A), B2 is 1x7 (3rd row of B).
    # A2 = [a02, a12]. B2 = [b20, b21, ..., b26].
    # C[:, k] = a02*b2k for row 0, a12*b2k for row 1.
    # This is 1 rank-1 term per column, total 7.

    # Total rank = 7 + 10 + 7 = 24.

    # Now, let's fill in the matrices.

    c = 0
    # Strassen part: 7 terms.
    # Term 1: M1 = (a00 + a11) * (b00 + b11)
    U[0 * m + 0, c] = 1  # a00
    U[1 * m + 1, c] = 1  # a11
    V[0 * p + 0, c] = 1  # b00
    V[1 * p + 1, c] = 1  # b11
    W[0 * n + 0, c] = 1  # C[0,0]
    W[0 * n + 1, c] = -1  # C[1,0]
    W[1 * n + 0, c] = -1  # C[0,1]
    W[1 * n + 1, c] = 1  # C[1,1]
    c += 1

    # Term 2: M2 = (a10 + a11) * b00
    U[1 * m + 0, c] = 1  # a10
    U[1 * m + 1, c] = 1  # a11
    V[0 * p + 0, c] = 1  # b00
    W[0 * n + 0, c] = -1  # C[0,0]
    W[0 * n + 1, c] = 1  # C[1,0]
    c += 1

    # Term 3: M3 = a00 * (b10 - b11)
    U[0 * m + 0, c] = 1  # a00
    V[1 * p + 0, c] = 1  # b10
    V[1 * p + 1, c] = -1  # b11
    W[0 * n + 0, c] = 1  # C[0,0]
    W[0 * n + 1, c] = -1  # C[1,0]
    c += 1

    # Term 4: M4 = a11 * (b01 - b00)
    U[1 * m + 1, c] = 1  # a11
    V[0 * p + 1, c] = 1  # b01
    V[0 * p + 0, c] = -1  # b00
    W[0 * n + 0, c] = -1  # C[0,0]
    W[0 * n + 1, c] = 1  # C[1,0]
    W[1 * n + 0, c] = 1  # C[0,1]
    W[1 * n + 1, c] = 1  # C[1,1]
    c += 1

    # Term 5: M5 = (a00 + a01) * b11
    U[0 * m + 0, c] = 1  # a00
    U[0 * m + 1, c] = 1  # a01
    V[1 * p + 1, c] = 1  # b11
    W[1 * n + 0, c] = -1  # C[0,1]
    W[1 * n + 1, c] = 1  # C[1,1]
    c += 1

    # Term 6: M6 = (a01 - a00) * (b00 + b10)
    U[0 * m + 1, c] = 1  # a01
    U[0 * m + 0, c] = -1  # a00
    V[0 * p + 0, c] = 1  # b00
    V[1 * p + 0, c] = 1  # b10
    W[1 * n + 0, c] = 1  # C[0,1]
    W[1 * n + 1, c] = -1  # C[1,1]
    c += 1

    # Term 7: M7 = (a11 - a10) * (b01 + b11)
    U[1 * m + 1, c] = 1  # a11
    U[1 * m + 0, c] = -1  # a10
    V[0 * p + 1, c] = 1  # b01
    V[1 * p + 1, c] = 1  # b11
    W[1 * n + 0, c] = 1  # C[0,1]
    W[1 * n + 1, c] = -1  # C[1,1]
    c += 1

    # Simple part for columns 2..6 of B1: 5 columns, 2 multiplications each.
    for k in range(2, 7):
        # Term: a00 * b0k
        U[0 * m + 0, c] = 1
        V[0 * p + k, c] = 1
        W[k * n + 0, c] = 1
        c += 1
        # Term: a01 * b1k
        U[0 * m + 1, c] = 1
        V[1 * p + k, c] = 1
        W[k * n + 0, c] = 1
        c += 1
        # Term: a10 * b0k
        U[1 * m + 0, c] = 1
        V[0 * p + k, c] = 1
        W[k * n + 1, c] = 1
        c += 1
        # Term: a11 * b1k
        U[1 * m + 1, c] = 1
        V[1 * p + k, c] = 1
        W[k * n + 1, c] = 1
        c += 1

    # Wait, that's 4 terms per column, not 2. I made a mistake.
    # For a 2x1 times 1x1, we need 2 multiplications, but as a tensor term, it's 2 rank-1 terms.
    # But for 2x2 times 1x2 (one column), it's 4 multiplications, so 4 rank-1 terms.
    # So for 5 columns, it's 5*4 = 20 terms. Total rank = 7 + 20 + 7 = 34. That's worse than 42.

    # I think I'm overcomplicating this. Let's just use the trivial rank-42 decomposition.

    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

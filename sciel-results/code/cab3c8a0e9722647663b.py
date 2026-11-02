"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# Standard Strassen 7-term coefficients for 2x2x2
# Each term: (u_coeffs, v_coeffs, w_coeffs)
# u: (A00, A01, A10, A11), v: (B00, B01, B10, B11), w: (C00, C01, C10, C11)
STRASSEN_TERMS = [
    # T1: (a+d)(e+h) -> C00+=1, C11+=1
    ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
    # T2: (c+d)e -> C10+=1, C11-=1
    ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
    # T3: a(f-h) -> C01+=1, C11+=1
    ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
    # T4: d(g-e) -> C00+=1, C10+=1
    ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),
    # T5: (a+b)h -> C00-=1, C01+=1
    ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
    # T6: (c-a)(e+f) -> C11+=1
    ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
    # T7: (b-d)(g+h) -> C00+=1
    ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    # Strategy:
    # Split m=5 into 1 + 4.
    # Part 1: 2x1x6 -> Rank 12 (standard)
    # Part 2: 2x4x6 -> Split m=4 into 2+2.
    #         2 * Rank(2x2x6)
    #         Rank(2x2x6) -> Split p=6 into 2+2+2.
    #         3 * Rank(2x2x2)
    #         Rank(2x2x2) -> Strassen Rank 7
    # Total Rank = 12 + 2 * 3 * 7 = 12 + 42 = 54.
    
    rank = 54
    
    # Initialize matrices as lists of lists
    # U is (n*m, rank) = (10, 54)
    # V is (m*p, rank) = (30, 54)
    # W is (p*n, rank) = (12, 54)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x6
    # A rows 0-1, cols 0-0. B rows 0-0, cols 0-5.
    # C rows 0-1, cols 0-5.
    for i in range(2):
        for k in range(6):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: 2x4x6
    # Split into two 2x2x6 blocks.
    # Block 1: j in [1, 2]
    # Block 2: j in [3, 4]
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        # For each 2x2x6 block, split p=6 into three 2x2x2 blocks.
        # k in [0,1], [2,3], [4,5]
        for s in range(3):
            k_base = s * 2
            # Apply 7 Strassen terms
            for (u_coeffs, v_coeffs, w_coeffs) in STRASSEN_TERMS:
                # u_coeffs: (A00, A01, A10, A11) -> indices in U
                # A[li, lj] is at U[(i_off+li)*m + (j_off+lj)]
                for li in range(2):
                    for lj in range(2):
                        val = u_coeffs[li * 2 + lj]
                        if val != 0:
                            gi = 0 + li  # i_off = 0
                            gj = j_base + lj
                            U[gi * m + gj][col] += val
                # v_coeffs: (B00, B01, B10, B11) -> indices in V
                # B[lj, lk] is at V[(j_off+lj)*p + (k_off+lk)]
                for lj in range(2):
                    for lk in range(2):
                        val = v_coeffs[lj * 2 + lk]
                        if val != 0:
                            gj = j_base + lj
                            gk = k_base + lk
                            V[gj * p + gk][col] += val
                # w_coeffs: (C00, C01, C10, C11) -> indices in W
                # C[li, lk] is at W[(k_off+lk)*n + (i_off+li)]
                for li in range(2):
                    for lk in range(2):
                        val = w_coeffs[li * 2 + lk]
                        if val != 0:
                            gi = 0 + li  # i_off = 0
                            gk = k_base + lk
                            W[gk * n + gi][col] += val
                col += 1
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

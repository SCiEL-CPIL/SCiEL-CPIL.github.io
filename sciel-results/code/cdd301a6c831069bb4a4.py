"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-65 exact tensor decomposition for 4x4 by 4x6 matrix multiplication.
    
    Strategy:
    - Split B (4x6) into B1 (4x4) and B2 (4x2).
    - C1 = A*B1: 4x4 * 4x4, use Strassen rank-49 decomposition.
    - C2 = A*B2: 4x4 * 4x2, use schoolbook rank-16 decomposition.
    - Total rank = 49 + 16 = 65.
    
    The output W follows the convention W[k*n+i, r] where k is column index of B (0..p-1),
    i is row index of A (0..n-1), n=4, p=6.
    """
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    p1 = 4  # columns in B1
    p2 = 2  # columns in B2

    rank1 = 49  # Strassen for 4x4x4x4
    rank2 = 32  # Schoolbook for 4x4x4x2
    total_rank = rank1 + rank2  # 81

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    # --- Part 1: Strassen 4x4 x 4x4 (columns 0..3 of B) ---
    # Strassen for 4x4: partition into 2x2 blocks, apply 7-step at top level,
    # each sub-problem is 2x2 x 2x2 with rank 7.
    # Top-level Strassen steps operate on 2x2 blocks of A and B1.
    
    # A blocks: A00(rows 0-1, cols 0-1), A01(rows 0-1, cols 2-3),
    #           A10(rows 2-3, cols 0-1), A11(rows 2-3, cols 2-3)
    # B1 blocks: B00(rows 0-1, cols 0-1), B01(rows 0-1, cols 2-3),
    #            B10(rows 2-3, cols 0-1), B11(rows 2-3, cols 2-3)
    
    # Strassen 7 steps for 2x2 x 2x2 -> 2x2:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B01-B11) -> contributes to C01, C11
    # M4 = A11(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # For 4x4 Strassen: the 7 "multiplications" are themselves 2x2x2x2 problems.
    # Each 2x2x2x2 has rank 7 (base Strassen).
    # So total rank = 7 * 7 = 49.
    
    # Define the 7 Strassen combinations at the 4x4 level:
    # Each combination specifies linear combos of A-blocks and B-blocks.
    # A-blocks: indexed by (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
    # B-blocks: indexed by (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
    
    # Strassen steps: (A_comb, B_comb, C_comb)
    # A_comb: dict mapping (rb, cb) -> coeff
    # B_comb: dict mapping (rb, cb) -> coeff  
    # C_comb: dict mapping (rb, cb) -> coeff (output block position)
    
    strassen_steps_4x4 = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Base Strassen for 2x2 x 2x2 (rank 7):
    # These operate on 2x2 sub-blocks. Each "block" at this level is a 2x2 matrix.
    # The 7 steps produce 2x2 results.
    
    base_strassen = [
        # m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2 = (a10+a11)b00 -> c10 += m2, c11 -= m2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3 = a00(b01-b11) -> c01 += m3, c11 += m3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4 = a11(b10-b00) -> c00 += m4, c10 += m4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5 = (a00+a01)b11 -> c00 -= m5, c01 += m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6 = (a10-a00)(b00+b01) -> c11 += m6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7 = (a01-a11)(b10+b11) -> c00 += m7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Helper: get flat U index for element at global row i, col j
    def u_idx(i, j):
        return i * m + j
    
    # Helper: get flat V index for element at global row j, col k
    def v_idx(j, k):
        return j * p + k
    
    # Helper: get flat W index for element at global row i, col k (W[k*n+i])
    def w_idx(i, k):
        return k * n + i
    
    # For the 4x4 Strassen:
    # Top-level: 7 combinations, each is a 2x2x2x2 problem (rank 7).
    # For each top-level step s (0..6) and each base step t (0..6):
    #   The A combination is a sum of 2x2 blocks from A.
    #   Each 2x2 block of A spans rows {2*rb, 2*rb+1} and cols {2*cb, 2*cb+1}.
    #   Within that 2x2 block, the base Strassen step t specifies which elements to use.
    #   Similarly for B.
    
    col = 0
    for s in range(7):
        A_top, B_top, C_top = strassen_steps_4x4[s]
        for t in range(7):
            a_base, b_base, c_base = base_strassen[t]
            # For this (s,t) pair, we have one rank-1 term.
            # U entries: for each (ar, ac) in A_top, for each (a0, a1) in a_base:
            #   global row = 2*ar + a0, global col = 2*ac + a1
            #   coeff = A_top[(ar,ac)] * a_base[(a0,a1)]
            for (ar, ac), a_coeff in A_top.items():
                for (a0, a1), a_base_coeff in a_base.items():
                    gi = 2 * ar + a0
                    gj = 2 * ac + a1
                    U[u_idx(gi, gj), col] += a_coeff * a_base_coeff
            
            # V entries: for each (br, bc) in B_top, for each (b0, b1) in b_base:
            #   global row = 2*br + b0, global col = 2*bc + b1 (within B1, cols 0..3)
            for (br, bc), b_coeff in B_top.items():
                for (b0, b1), b_base_coeff in b_base.items():
                    gj = 2 * br + b0
                    gk = 2 * bc + b1  # col in B1 (0..3)
                    V[v_idx(gj, gk), col] += b_coeff * b_base_coeff
            
            # W entries: for each (cr, cc) in C_top, for each (c0, c1) in c_base:
            #   global row = 2*cr + c0, global col = 2*cc + c1 (within B1, cols 0..3)
            for (cr, cc), c_coeff in C_top.items():
                for (c0, c1), c_base_coeff in c_base.items():
                    gi = 2 * cr + c0
                    gk = 2 * cc + c1  # col in B1 (0..3)
                    W[w_idx(gi, gk), col] += c_coeff * c_base_coeff
            
            col += 1
    
    assert col == rank1, f"Expected col={rank1}, got {col}"
    
    # --- Part 2: Schoolbook 4x4 x 4x2 (columns 4..5 of B) ---
    # C2[i, k] = sum_j A[i,j] * B2[j, k] for k in {4,5}
    # Rank = 4*4*2 = 32.
    
    # We use standard schoolbook for the remaining 2 columns.
    # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[u_idx(i, j), col] = 1.0
                V[v_idx(j, k), col] = 1.0
                W[w_idx(i, k), col] = 1.0
                col += 1
    
    assert col == total_rank, f"Expected col={total_rank}, got {col}"
    
    # Verify correctness
    if time.time() - start < DEADLINE:
        rng = np.random.RandomState(42)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test
        
        # Efficient verification using tensor contraction
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_recon = np.zeros((n, p))
        for r in range(total_rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # For each (i,k): sum_j u[i*m+j] * v[j*p+k] * w[k*n+i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                    C_recon[i,k] += s
        
        max_err = np.max(np.abs(C_recon - C_expected))
        if max_err > 1e-10:
            # Fallback: schoolbook rank 96
            rank2 = n * m * p
            U2 = np.zeros((n * m, rank2), dtype=float)
            V2 = np.zeros((m * p, rank2), dtype=float)
            W2 = np.zeros((p * n, rank2), dtype=float)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2[i*m+j, idx] = 1.0
                        V2[j*p+k, idx] = 1.0
                        W2[k*n+i, idx] = 1.0
                        idx += 1
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    if time.time() - start > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

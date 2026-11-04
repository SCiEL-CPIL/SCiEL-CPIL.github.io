"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i (trace convention)
    # C[0,0]: i=0,k=0 -> W[0]
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C[0,1]: i=0,k=1 -> W[2]
    W[2, 2] = 1; W[2, 4] = 1
    # C[1,0]: i=1,k=0 -> W[1]
    W[1, 1] = 1; W[1, 3] = 1
    # C[1,1]: i=1,k=1 -> W[3]
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _decompose_4x4_x_4x6():
    """
    Build the full decomposition using a 2x2 block structure.
    A (4x4) is 2x2 blocks of size 2x2.
    B (4x6) is 2x3 blocks of size 2x2.
    C (4x6) is 2x3 blocks of size 2x2.
    
    For each pair (outer_a, outer_b) we need to compute
    sum over inner blocks of A_block * B_block -> C_block.
    
    Strategy:
    - For the first 2 columns of B (ck=0,1): use Strassen on the 2x2*2x2 block products.
      The outer structure for 2x2 (A-blocks) * 2x2 (B-blocks for cols 0,1) -> 2x2 (C-blocks for cols 0,1)
      is itself a 2x2 * 2x2 -> 2x2 problem, which can be done with rank 7 (Strassen).
      So for ck in {0,1}, we use 7 outer terms, each with inner Strassen (7) = 49 terms.
      But wait, the outer structure combines A[ai][bj] * B[bj][ck] for fixed ck.
      Actually, for fixed ck, we compute C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck].
      This is a 2x2 * 2x2 -> 2x2 matrix multiplication (in the block indices ai, bj, ck).
      So rank 7 for the outer, times 7 for inner = 49 per ck.
      For 2 such ck values: 98.
      For ck=2: schoolbook 4 outer * 4 inner = 16.
      Total: 98 + 16 = 114. That's worse than 77.
    
    Better strategy (as in current code):
    - For cols 0,1 of B: the outer 2x2*2x2->2x2 is done with Strassen (7 terms).
      Each term picks certain A-blocks and B-blocks. The B-blocks are 2x2 (for cols 0,1).
      Inner: each A-block (2x2) * B-block (2x2) -> C-block (2x2) uses Strassen (7).
      Total for cols 0,1: 7 * 7 = 49.
    - For col 2 of B: the B-block is 2x2 (for col 2 only).
      C[ai][2] = sum_bj A[ai][bj] * B[bj][2].
      This is 2x2 * 2x2 -> 2x2 in block indices, but B only has one column of blocks.
      Schoolbook: 4 terms (ai=0,1; bj=0,1).
      Each inner: A-block (2x2) * B-block (2x2) -> C-block (2x2) uses Strassen (7).
      Total for col 2: 4 * 7 = 28.
    - Grand total: 49 + 28 = 77.
    """
    sU, sV, sW = _strassen_2x2()
    
    n, m, p = 4, 4, 6
    total_rank = 77
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    col_idx = 0
    
    # Part 1: cols 0,1 of B (ck=0,1), using Strassen outer (7 terms)
    # Outer Strassen for block indices:
    # A-blocks indexed by (ai, bj) in {0,1}x{0,1}
    # B-blocks indexed by (bj, ck) in {0,1}x{0,1}  [ck=0,1]
    # C-blocks indexed by (ai, ck) in {0,1}x{0,1}  [ck=0,1]
    
    for t in range(7):
        for t_inner in range(7):
            c = col_idx + t_inner
            # U part
            for (ai, bj), coeff_outer in _get_outer_U_strassen(t).items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t_inner]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, c] += coeff_outer * coeff_inner
            # V part
            for (bj, ck), coeff_outer in _get_outer_V_strassen(t).items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t_inner]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, c] += coeff_outer * coeff_inner
            # W part
            for (ai, ck), coeff_outer in _get_outer_W_strassen(t).items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t_inner]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, c] += coeff_outer * coeff_inner
    col_idx += 49
    
    # Part 2: col 2 of B (ck=2), schoolbook outer (4 terms)
    for a in range(2):
        for b in range(2):
            for t_inner in range(7):
                c = col_idx + t_inner
                # U part: A-block (a,b)
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t_inner]
                        if coeff_inner != 0:
                            u_row = (2*a + ai2) * 4 + (2*b + bj2)
                            U[u_row, c] += coeff_inner
                # V part: B-block (b, 2)
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t_inner]
                        if coeff_inner != 0:
                            v_row = (2*b + bj2) * 6 + (2*2 + ck2)
                            V[v_row, c] += coeff_inner
                # W part: C-block (a, 2)
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t_inner]
                        if coeff_inner != 0:
                            w_row = (2*2 + ck2) * 4 + (2*a + ai2)
                            W[w_row, c] += coeff_inner
            col_idx += 7
    
    return U, V, W


def _get_outer_U_strassen(t):
    """Return dict for U part of outer Strassen term t."""
    # Same as sU but for block indices
    d = {}
    sU, _, _ = _strassen_2x2()
    for ai in range(2):
        for bj in range(2):
            coeff = sU[ai*2 + bj, t]
            if coeff != 0:
                d[(ai, bj)] = coeff
    return d


def _get_outer_V_strassen(t):
    """Return dict for V part of outer Strassen term t."""
    d = {}
    _, sV, _ = _strassen_2x2()
    for bj in range(2):
        for ck in range(2):
            coeff = sV[bj*2 + ck, t]
            if coeff != 0:
                d[(bj, ck)] = coeff
    return d


def _get_outer_W_strassen(t):
    """Return dict for W part of outer Strassen term t."""
    d = {}
    _, _, sW = _strassen_2x2()
    for ai in range(2):
        for ck in range(2):
            coeff = sW[ck*2 + ai, t]
            if coeff != 0:
                d[(ai, ck)] = coeff
    return d


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    U, V, W = _decompose_4x4_x_4x6()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

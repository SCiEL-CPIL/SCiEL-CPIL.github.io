"""Search for a rank-minimal exact decomposition of the 5x5x5 matrix multiplication tensor.

Strategy:
  1. Construct a structured low-rank decomposition using block matrix multiplication
     with Strassen's 7-multiplication algorithm for 2x2 blocks.
  2. Split 5 = 2 + 2 + 1 and use Strassen for the 2x2 block multiplications.
  3. Verify the decomposition exactly; fall back to schoolbook if verification fails.

The output follows the trace convention W[k*n+i, r].
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]
N = 5
TIME_LIMIT = 140.0  # safety margin below the 150s wall-clock kill


def _schoolbook():
    """Return the guaranteed-valid schoolbook rank-125 decomposition."""
    n, m, p = DIMENSIONS
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


def _build_target():
    """T[i,j,k] = 1 if i==k else 0, for n=m=p=5. Shape (5,5,5)."""
    T = np.zeros((N, N, N), dtype=np.int64)
    for i in range(N):
        for k in range(N):
            if i == k:
                for j in range(N):
                    T[i, j, k] = 1
    return T


def _verify(U, V, W, T):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == T[i,j,k] for all i,j,k."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = int(np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i]))
                if val != T[i, j, k]:
                    return False
    return True








def _block_decomposition_v2():
    """
    Construct a block-based decomposition for 5x5 matrix multiplication.
    Split into blocks of size (2, 2, 1).
    Uses Strassen for 2x2x2 block multiplications, standard for others.
    """
    n = N
    block_sizes = [2, 2, 1]
    
    block_ranges = []
    start = 0
    for bs in block_sizes:
        block_ranges.append((start, start + bs))
        start += bs
    
    U_entries = []
    V_entries = []
    W_entries = []
    col = 0
    
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                ai = block_sizes[i_block]
                aj = block_sizes[j_block]
                bk = block_sizes[k_block]
                
                ai_start = block_ranges[i_block][0]
                aj_start = block_ranges[j_block][0]
                bk_start = block_ranges[k_block][0]
                
                if ai == 2 and aj == 2 and bk == 2:
                    # Strassen 2x2x2: 7 multiplications
                    # A = [[A00, A01], [A10, A11]]
                    # B = [[B00, B01], [B10, B11]]
                    # C = A*B
                    
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B01-B11)
                    # M4 = A11(B10-B00)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    
                    # C00 = M1 - M4 + M5 - M7
                    # C01 = M3 + M4
                    # C10 = M2 + M5
                    # C11 = M1 - M2 + M3 - M6
                    
                    M_defs = [
                        # (U_coeffs, V_coeffs) where keys are (row, col) within block
                        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
                        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),                # M2
                        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),               # M3
                        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),               # M4
                        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),                # M5
                        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),     # M6
                        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),     # M7
                    ]
                    
                    C_coeffs = [
                        [1, 0, 0, 0, 0, 0, 0],   # placeholder
                    ]
                    # C00 = M1 - M4 + M5 - M7: coeffs for M1..M7 = [1, 0, 0, -1, 1, 0, -1]
                    # C01 = M3 + M4: [0, 0, 1, 1, 0, 0, 0]
                    # C10 = M2 + M5: [0, 1, 0, 0, 1, 0, 0]
                    # C11 = M1 - M2 + M3 - M6: [1, -1, 1, 0, 0, -1, 0]
                    
                    C_out_coeffs = {
                        (0, 0): [1, 0, 0, -1, 1, 0, -1],
                        (0, 1): [0, 0, 1, 1, 0, 0, 0],
                        (1, 0): [0, 1, 0, 0, 1, 0, 0],
                        (1, 1): [1, -1, 1, 0, 0, -1, 0],
                    }
                    
                    strassen_start_col = col
                    for m_idx, (U_comb, V_comb) in enumerate(M_defs):
                        for (br, bc), coeff in U_comb.items():
                            r = ai_start + br
                            c = aj_start + bc
                            U_entries.append((r * n + c, col, coeff))
                        for (br, bc), coeff in V_comb.items():
                            r = aj_start + br
                            c = bk_start + bc
                            V_entries.append((r * n + c, col, coeff))
                        col += 1
                    
                    # Add W entries
                    for (cr, cc), coeffs in C_out_coeffs.items():
                        for m_idx, coeff in enumerate(coeffs):
                            if coeff != 0:
                                r = ai_start + cr
                                c = bk_start + cc
                                W_entries.append((c * n + r, strassen_start_col + m_idx, coeff))
                
                else:
                    # Standard multiplication
                    for r in range(ai):
                        for s in range(aj):
                            for t in range(bk):
                                ai_r = ai_start + r
                                aj_s = aj_start + s
                                bk_t = bk_start + t
                                
                                U_entries.append((ai_r * n + aj_s, col, 1))
                                V_entries.append((aj_s * n + bk_t, col, 1))
                                W_entries.append((bk_t * n + ai_r, col, 1))
                                col += 1
    
    rank = col
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    for r, c, v in U_entries:
        U[r][c] += v
    for r, c, v in V_entries:
        V[r][c] += v
    for r, c, v in W_entries:
        W[r][c] += v
    
    return {"U": U, "V": V, "W": W}, rank


def _verify_decomposition(decomp, T):
    """Verify a decomposition against the target tensor."""
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    return _verify(U, V, W, T)


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT
    T = _build_target()
    
    # Try the block-based Strassen decomposition
    decomp, rank = _block_decomposition_v2()
    if _verify_decomposition(decomp, T):
        return decomp
    
    # Fallback: schoolbook
    return _schoolbook()
